"""Free-boundary equilibria by virtual casing, with the boundary as the unknown.

At a free boundary ``S`` without a sheet current the plasma-vacuum interface conditions are
(Conlin et al. 2024, arXiv:2412.05680)

1. ``B_out . n = 0``,
2. ``|B_out|^2 = |B_in|^2 + 2 mu0 p``,
3. ``n x (B_out - B_in) = mu0 K = 0``,

with ``B_out = B_coil + B_plasma`` and ``B_plasma`` the plasma's own field from the virtual-casing principle
(:mod:`vmex.core.virtual_casing`).  Two curl- and divergence-free fields outside ``S`` with the same normal
component and the same net poloidal current ``G`` are equal, so (1) and ``G_plasma = G_coil`` imply (2) and (3)
when the edge pressure vanishes (a nonzero edge pressure needs a sheet current and is rejected).  (2) and (3) are
therefore reported (:func:`boundary_residual`), not fitted.

The unknown is the displacement ``h(theta, zeta)`` of the deck's boundary along the deck equilibrium's radial
coordinate lines, measured normal to the surface (:func:`_normal_displacement_map`).  The deck fixes the
boundary's poloidal labelling, so the answer does not depend on the start; ``initial_boundary`` only sets the
starting ``h``.  The equations are square in ``h``, with the row ``G_plasma = G_coil`` for the ``(0, 0)`` mode:

- vacuum (no pressure, no plasma current): ``B_out = B_coil`` is an explicit function of the boundary, and the
  rows are the stationarity of the area-weighted mean square of ``B_coil . n`` in ``h``
  (:func:`_mean_square_normal_field`), with its exact Hessian as the Newton Jacobian.  ``h`` also carries the
  poloidal harmonic ``m = mpol``, where the ``B . n`` left by the top boundary modes lies;
- finite beta: Petrov-Galerkin rows ``T0^T r = 0`` for the pointwise residual ``r = sqrt(w) B_out . n / |B_in|``
  (``w`` the area weights), whose test functions ``T0`` are the exact response of the deck's ``r`` to each
  boundary coordinate, the plasma's own response included (:meth:`ThreeTermsFreeBoundaryModel._deck_response`).
  Test functions without the plasma response are weakest on the near-resonant modes, where the plasma responds
  most; test functions at the solution itself would make the design derivatives need second derivatives of the
  equilibrium.

The boundary's band is the interior's, ``max(mpol - 1, ntor)``.  Where Newton on the band stalls, the band is
built up from ``m <= 1`` one mode at a time (:func:`solve_band_continuation`).  Every trial boundary is a
fixed-boundary equilibrium (Newton-refined; :mod:`vmex.core.implicit`), so the rows and their implicit derivatives
are smooth functions of the boundary, the external field and the plasma parameters.  VMEC2000 counterpart: none;
this replaces NESTOR (:mod:`vmex.core.vacuum`) as the free-boundary condition.

Entry points, from the most to the least packaged:

- a free-boundary deck with ``!@VMEX BOUNDARY_CONDITION = THREE_TERMS`` (``vmex input.case``), the CLI flag
  ``--boundary-condition three-terms``, or ``solve_free_boundary_multigrid(..., boundary_condition="three_terms")``;
- :func:`solve_free_boundary_three_terms`: one solve with every control, and ``previous=`` to restart after the
  external field changes;
- ``FreeBoundaryProblem.from_loss(..., boundary_condition="three_terms")`` (:class:`ThreeTermsFreeBoundaryProblem`):
  single-stage optimization of a loss and constraint rows of coils and plasma parameters at their free boundary;
- :class:`ThreeTermsFreeBoundaryModel`: the rows, their Jacobian, state tangents and pullbacks for one deck and
  any external field or plasma parameters, without recompiling;
- :func:`boundary_residual`: the three conditions on any equilibrium, e.g. a NESTOR result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import hashlib
import time
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from . import virtual_casing as vc
from .errors import TrialRejected, VmecConvergenceError, VmecError
from .input import VmecInput
from .loops import map_columns
from .problem import FunctionProblem
from .profiles import MU0
from .solver import EarlyStop


@dataclass(frozen=True)
class BoundaryResidual:
    """The interface conditions on a boundary, pointwise relative to ``|B|`` there.

    ``normal`` is the area-weighted RMS of ``|B_out . n| / |B_in|`` over the boundary (``n`` the unit normal) and
    ``normal_max`` its maximum; ``pressure`` is the area-weighted RMS of
    ``(|B_out|^2 - |B_in|^2 - 2 mu0 p) / (2 |B_in|^2)`` and ``sheet_current`` that of
    ``|n x (B_out - B_in)| / |B_in| = mu0 |K| / |B_in|``.  In vacuum (no plasma current or pressure) the field
    outside is the external field alone and the only condition is ``B . n = 0``: ``pressure`` and
    ``sheet_current`` are ``None``.  ``|B|`` is the interior field's (equal to the outside field's where the
    conditions hold).  One definition for every code: DESC
    prints a different normalization (the mean of ``|B . n|`` over ``B_s R0 a``, ``BoundaryError``), so
    comparisons evaluate every code's boundary with this one.
    """

    normal: float
    normal_max: float
    pressure: float | None = None
    sheet_current: float | None = None

    def __str__(self) -> str:
        text = f"B.n/|B| RMS {self.normal:.3e} max {self.normal_max:.3e}"
        if self.pressure is not None:
            text += f", pressure balance RMS {self.pressure:.3e}, sheet current RMS {self.sheet_current:.3e}"
        return text


@dataclass(frozen=True)
class _DeckField:
    """An external field with the deck's response data (:meth:`ThreeTermsFreeBoundaryModel._deck_response`).

    The finite-beta rows' test functions combine the field with these deck arrays; they travel with the field as
    one pytree argument of the compiled rows, not as constants of the programs
    (:meth:`ThreeTermsFreeBoundaryModel.attach`).
    """

    field: Any
    deck: Any


jax.tree_util.register_dataclass(_DeckField, data_fields=["field", "deck"], meta_fields=[])


def _fingerprint(*trees) -> str:
    """Hash of the concrete arrays of ``trees`` (cache keys of linearizations)."""
    digest = hashlib.sha1()
    for leaf in jax.tree.leaves(trees):
        digest.update(np.asarray(leaf).tobytes())
    return digest.hexdigest()


def _interface_terms(inp, state, external_field, *, runtime, nphi, ntheta, digits, precision, p_edge):
    """Rows and weights of :func:`boundary_residual`, plus the boundary points ``(3, nphi, ntheta)``.

    ``precision=None`` with ``digits=None`` is the vacuum: no plasma current, so ``B_out`` is the external field
    alone (no virtual casing, whose quadrature would only add its error), and ``B . n = 0`` is the only
    condition: one row, ``B_out . n / |B_in|``.
    """
    data = vc.surface_field_data_from_state(inp, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
    if digits is None:
        B_out = vc.external_B_cartesian(external_field, jnp.asarray(data.gamma), jnp.asarray(data.phi))
        area = jnp.linalg.norm(jnp.asarray(data.area_vector), axis=0)
        normal = jnp.sum(B_out * jnp.asarray(data.normal), axis=0) / jnp.linalg.norm(jnp.asarray(data.B_total), axis=0)
        return normal[None], area / jnp.sum(area), data.gamma
    interface = vc.PlasmaVacuumInterface.from_surface_data(
        data, p_edge=p_edge, digits=digits, precision=precision)
    B_in = jnp.asarray(data.B_total)
    B_out = interface.total_B_out(external_field)
    B2 = interface.Bin_mag2
    scale = jnp.sqrt(B2)
    normal = jnp.sum(B_out * interface.normal, axis=0) / scale
    pressure = interface.pressure_balance_residual(external_field) / (2.0 * B2)
    sheet = jnp.cross(interface.normal, B_out - B_in, axis=0) / scale
    return jnp.concatenate([normal[None], pressure[None], sheet]), interface.weights, interface.gamma


def boundary_residual(inp: VmecInput, state, external_field: Any, *, runtime=None,
                      nphi: int, ntheta: int, digits: int = vc.VC_DIGITS, precision=None,
                      p_edge: float = 0.0):
    """Pointwise interface conditions on the boundary of a live ``state``.

    Returns ``(rows, weights)``: ``rows`` has shape ``(5, nphi, ntheta)`` with
    ``B_out . n``, the pressure-balance jump and the three Cartesian components
    of ``n x (B_out - B_in)``, each scaled as in :class:`BoundaryResidual` --
    for a vacuum deck (no plasma current or pressure, :func:`_is_vacuum`) the
    single row ``B_out . n / |B_in|`` with ``B_out`` the external field;
    ``weights`` are the area weights (summing to one).  Traceable in ``state``
    and the external-field parameters when ``precision`` is supplied (see
    :func:`~vmex.core.virtual_casing.plan_vc_precision`).
    """
    if _is_vacuum(inp):
        digits, precision = None, None
    rows, weights, _ = _interface_terms(inp, state, external_field, runtime=runtime, nphi=nphi,
                                        ntheta=ntheta, digits=digits, precision=precision, p_edge=p_edge)
    return rows, weights


def summarize_boundary_residual(rows, weights) -> BoundaryResidual:
    """:class:`BoundaryResidual` of :func:`boundary_residual`'s output (one row in vacuum: ``B . n`` alone)."""
    rows, weights = np.asarray(rows), np.asarray(weights)

    def rms(value):
        return float(np.sqrt(np.sum(weights * value**2)))

    normal = dict(normal=rms(rows[0]), normal_max=float(np.max(np.abs(rows[0]))))
    if rows.shape[0] == 1:
        return BoundaryResidual(**normal)
    return BoundaryResidual(**normal, pressure=rms(rows[1]), sheet_current=rms(np.linalg.norm(rows[2:], axis=0)))


def _is_vacuum(inp: VmecInput) -> bool:
    """No pressure and no plasma current that the field can feel: ``B_out`` is the external field alone.

    A current or pressure below the field's round-off (a CURTOR of 1e-12 A read back from a WOUT) cannot change
    ``B`` in double precision, so it does not select the finite-beta formulation.
    """
    return (int(inp.ncurr) == 1 and _plasma_below_roundoff(inp)
            and (inp.ac is None or not np.any(np.asarray(inp.ac, dtype=float))))


def _edge_pressure(inp: VmecInput) -> tuple[float, float]:
    """Edge and peak pressure of ``inp`` in Pa."""
    from .profiles import pressure

    p = np.asarray(pressure(inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, np.linspace(0.0, 1.0, 101),
                            pres_scale=inp.pres_scale, bloat=inp.bloat, spres_ped=inp.spres_ped))
    return float(p[-1]), float(np.max(np.abs(p)))


def _plasma_below_roundoff(inp: VmecInput) -> bool:
    """Whether the plasma terms of ``inp`` (its total current ``CURTOR`` and its peak pressure) are below the
    round-off of its magnetic field, so that they cannot change ``B`` in double precision.

    The field's scale is the deck's toroidal field ``B_t = |PHIEDGE| / A`` (``A`` the area of the boundary's
    ``phi = 0`` cross section, minor radius ``a = sqrt(A / pi)``); the current's field at the edge is
    ``mu0 |I| / (2 pi a)`` and the pressure's is ``sqrt(2 mu0 p)``.  Each must be at most ``eps B_t``, machine
    epsilon: below that adding it to ``B_t`` leaves ``B_t`` unchanged, so it is zero for the solver.
    """
    rbc, zbs = np.asarray(inp.rbc, dtype=float), np.asarray(inp.zbs, dtype=float)
    m = np.arange(rbc.shape[1])
    area = abs(np.pi * np.sum(m * rbc.sum(axis=0) * zbs.sum(axis=0)))  # |closed integral of R dZ| at phi = 0
    if not area > 0.0:
        return False
    B_t, eps = abs(float(inp.phiedge)) / area, float(np.finfo(float).eps)
    B_current = MU0 * abs(float(inp.curtor)) / (2.0 * np.pi * np.sqrt(area / np.pi))
    B_pressure = np.sqrt(2.0 * MU0 * _edge_pressure(inp)[1])
    return bool(B_current <= eps * B_t and B_pressure <= eps * B_t)


def _check_edge_pressure(inp: VmecInput) -> None:
    """Reject a deck whose pressure does not vanish at the boundary: without a sheet current the interface
    conditions (module docstring) need ``p(1) = 0`` (to ``1e-8`` of the peak pressure)."""
    p_edge, p_max = _edge_pressure(inp)
    if abs(p_edge) > 1e-8 * p_max:
        raise ValueError(f"edge pressure is {p_edge:g} Pa: without a sheet current the "
                         "boundary pressure must vanish")


def _digest_into(h, obj):
    """Feed ``obj`` to the hash ``h``: arrays by dtype, shape and bytes; containers and dataclasses by content."""
    import dataclasses

    if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes)):
        h.update(repr((type(obj).__name__, obj)).encode())
    elif isinstance(obj, (np.ndarray, jax.Array, np.generic)):
        a = np.ascontiguousarray(np.asarray(obj))
        h.update(repr((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(type(obj).__qualname__.encode())
        for f in dataclasses.fields(obj):
            h.update(f.name.encode())
            _digest_into(h, getattr(obj, f.name))
    elif isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            _digest_into(h, k)
            _digest_into(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(f"{type(obj).__name__}{len(obj)}".encode())
        for item in obj:
            _digest_into(h, item)
    else:
        leaves, treedef = jax.tree.flatten(obj)
        if leaves and not (len(leaves) == 1 and leaves[0] is obj):
            h.update(repr(treedef).encode())
            for leaf in leaves:
                _digest_into(h, leaf)
        elif hasattr(obj, "__dict__"):
            h.update(type(obj).__qualname__.encode())
            _digest_into(h, dict(vars(obj)))
        else:
            h.update(repr(obj).encode())


def _digest(*objects) -> str:
    """A content hash of ``objects`` (:func:`_digest_into`)."""
    h = hashlib.sha256()
    for obj in objects:
        _digest_into(h, obj)
    return h.hexdigest()


_SOURCE_DIGEST: list[str] = []


def _source_digest() -> str:
    """A hash of the vmex package's Python sources, so cached data never outlive the code that made them."""
    if not _SOURCE_DIGEST:
        import pathlib

        h = hashlib.sha256()
        root = pathlib.Path(__file__).resolve().parents[1]
        for path in sorted(root.rglob("*.py")):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
        _SOURCE_DIGEST.append(h.hexdigest())
    return _SOURCE_DIGEST[0]


def _deck_cache_path(key: str):
    """``<persistent compilation cache>/deck_response/<key>.npz``, or ``None`` when that cache is off."""
    import pathlib

    from .. import _compat

    root = _compat._default_compilation_cache_dir()
    return None if root is None else pathlib.Path(root) / "deck_response" / f"{key}.npz"


def _deck_cache_load(key: str):
    """The cached ``(primal, tangent)`` deck response of ``key`` (each four arrays), or ``None``."""
    path = _deck_cache_path(key)
    if path is None or not path.exists():
        return None
    try:
        with np.load(path) as data:
            leaves = [jnp.asarray(data[f"a{i}"]) for i in range(8)]
    except (OSError, KeyError, ValueError):  # a damaged entry is a miss
        return None
    return tuple(leaves[:4]), tuple(leaves[4:])


def _deck_cache_store(key: str, value) -> None:
    """Write ``value`` atomically (a private temporary file renamed into place); a failed write is no error."""
    import contextlib
    import os
    import tempfile

    path = _deck_cache_path(key)
    leaves = jax.tree.leaves(value)
    if path is None or len(leaves) != 8:
        return
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **{f"a{i}": np.asarray(a) for i, a in enumerate(leaves)})
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _deck_cached(key: str, compute):
    """The cached deck response of ``key``, or ``compute()`` stored under it."""
    cached = _deck_cache_load(key)
    if cached is not None:
        return cached
    value = compute()
    _deck_cache_store(key, value)
    return value


def _axis_loop(wout):
    """Points ``(3, n)``, tangent and spacing of the magnetic axis of ``wout``: a closed curve inside the coils."""
    phi = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    harmonics = np.arange(np.asarray(wout.raxis_cc).size) * int(wout.nfp)
    rc, zs = np.asarray(wout.raxis_cc, dtype=float), np.asarray(wout.zaxis_cs, dtype=float)
    cos, sin = np.cos(np.outer(phi, harmonics)), np.sin(np.outer(phi, harmonics))
    R, Z = cos @ rc, -sin @ zs
    R_p, Z_p = -sin @ (rc * harmonics), -cos @ (zs * harmonics)  # exact tangent: the periodic sum is spectral
    points = np.stack([R * np.cos(phi), R * np.sin(phi), Z])
    tangent = np.stack([R_p * np.cos(phi) - R * np.sin(phi), R_p * np.sin(phi) + R * np.cos(phi), Z_p])
    return jnp.asarray(points), jnp.asarray(tangent), float(phi[1] - phi[0])


def _net_current(field, points, tangent, dphi):
    """Traceable ``G = (1/2 pi)`` circulation of ``field`` along a closed curve (:func:`_axis_loop`)."""
    B = vc.external_B_cartesian(field, points[:, :, None])[:, :, 0]
    return jnp.sum(B * tangent) * dphi / (2.0 * jnp.pi)


def _coil_net_current(external_field, wout) -> float:
    """``G = (1/2 pi)`` times the external field's circulation around the magnetic axis of ``wout``."""
    return float(_net_current(external_field, *_axis_loop(wout)))


def _surface_frame(rmnc, zmns, xm, xn, phi, theta):
    """Points, ``e_theta``, ``e_phi`` (each ``(3, nphi, ntheta)``) of the surface ``(rmnc, zmns)`` (traceable).

    ``phi, theta`` are the ``(nphi, ntheta)`` grid; the series is ``sum cos|sin(xm theta - xn phi)``.
    """
    phase = xm[:, None, None] * theta[None] - xn[:, None, None] * phi[None]
    cos, sin = jnp.cos(phase), jnp.sin(phase)
    R, Z = jnp.tensordot(rmnc, cos, 1), jnp.tensordot(zmns, sin, 1)
    R_t, Z_t = jnp.tensordot(-rmnc * xm, sin, 1), jnp.tensordot(zmns * xm, cos, 1)
    R_p, Z_p = jnp.tensordot(rmnc * xn, sin, 1), jnp.tensordot(-zmns * xn, cos, 1)
    c, s = jnp.cos(phi), jnp.sin(phi)
    return (jnp.stack([R * c, R * s, Z]), jnp.stack([R_t * c, R_t * s, Z_t]),
            jnp.stack([R_p * c - R * s, R_p * s + R * c, Z_p]))


def _mean_square_normal_field(rmnc, zmns, xm, xn, phi, theta, phi_grid, field):
    """Area-weighted mean of ``(B . n)^2`` of ``field`` on the surface ``(rmnc, zmns)`` (traceable).

    ``phi, theta`` are the ``(nphi, ntheta)`` grid (one field period), ``phi_grid`` its ``phi`` axis.
    """
    points, e_theta, e_phi = _surface_frame(rmnc, zmns, xm, xn, phi, theta)
    N = jnp.cross(e_theta, e_phi, axis=0)
    area = jnp.linalg.norm(N, axis=0)
    B = vc.external_B_cartesian(field, points, phi_grid)
    return jnp.sum(jnp.sum(B * N, axis=0) ** 2 / area) / jnp.sum(area)


#: Relative change of the displacement map under a doubled grid at which its grid counts as converged: the map's
#: error then moves a boundary displacement ``h`` by less than ``sqrt(eps) |h|``, below the Newton step tolerance
#: (:func:`newton_step_converged`).
_MAP_GRID_RTOL = float(np.sqrt(np.finfo(float).eps))


def _normal_displacement_map(inp, max_mode, alpha=1.2, ntheta=None, nzeta=None, direction=None, extend=False):
    """:func:`_normal_displacement_map_on` on a grid that resolves it: from four points per boundary harmonic in
    each direction (``4 mpol`` poloidal, ``4 ntor + 4`` per field period), doubled until a doubling changes the map
    by less than ``_MAP_GRID_RTOL`` of its largest entry.  The displacement times the direction field
    ``g = n / (n_R^2 + n_Z^2)`` is not band-limited, so the fit depends on the grid until it is resolved.
    ``ntheta, nzeta`` fix the grid instead (no convergence check)."""
    if ntheta is not None and nzeta is not None:
        return _normal_displacement_map_on(inp, max_mode, alpha, int(ntheta), int(nzeta), direction, extend)
    ntor, mpol = (np.shape(inp.rbc)[0] - 1) // 2, np.shape(inp.rbc)[1]
    nt, nz = 4 * mpol, 4 * ntor + 4
    out = _normal_displacement_map_on(inp, max_mode, alpha, nt, nz, direction, extend)
    while True:
        nt, nz = 2 * nt, 2 * nz
        finer = _normal_displacement_map_on(inp, max_mode, alpha, nt, nz, direction, extend)
        change = np.max(np.abs(finer[0] - out[0])) / np.max(np.abs(finer[0]))
        out = finer
        if change <= _MAP_GRID_RTOL:
            return out
        if nt * nz > 2**22:  # 4M points: the map does not converge (a non-smooth direction field)
            raise ValueError(f"the boundary displacement map does not converge in its grid ({change:.1e} at "
                             f"{nt} x {nz})")


def _normal_displacement_map_on(inp, max_mode, alpha, ntheta, nzeta, direction, extend=False):
    """Boundary coordinates as a normal displacement of ``inp``'s surface: ``(M, modes, x_scale)``.

    ``h(theta, zeta) = sum h_mn cos(m theta - n nfp zeta)`` over ``modes`` (those of
    :func:`~vmex.core.optimize.pack_boundary`, with ``extend`` the next poloidal harmonic ``m_top + 1`` on the
    same toroidal modes, then ``(0, 0)``) moves each boundary point along the surface normal, within its ``phi = zeta`` plane:
    ``(dR, dZ) = h (N_R, N_Z) / (N_R^2 + N_Z^2)`` with ``N`` the unit normal, so its normal displacement is
    ``h``, or along ``direction``.  Its truncated Fourier fit is linear in ``h``: the packed boundary
    coordinates are ``x0 + M h``.

    The harmonic ``m_top + 1`` has no coefficient of its own: through the poloidal variation of the transport
    direction it moves the top coefficients ``m_top``.  Without it the ``B . n`` that the boundary's top modes
    leave at ``m_top + 1`` is outside every row's reach; with it the vacuum least squares sees and lowers it.
    The remaining tangential moves (relabellings ``theta -> theta + delta(theta, zeta)``, which no interface
    condition sees) stay absent from the unknowns.
    """
    from .optimize import _dof_modes, pack_boundary

    rbc, zbs, nfp = np.asarray(inp.rbc, dtype=float), np.asarray(inp.zbs, dtype=float), int(inp.nfp)
    ntor, mpol = (rbc.shape[0] - 1) // 2, rbc.shape[1]
    th = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    ze = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False)
    T, Zg = np.meshgrid(th, ze, indexing="ij")
    ns, ms = np.arange(-ntor, ntor + 1), np.arange(mpol)
    ph = ms[None, :, None, None] * T[None, None] - ns[:, None, None, None] * nfp * Zg[None, None]
    cos, sin = np.cos(ph), np.sin(ph)
    del ph
    R = np.einsum("nm,nmij->ij", rbc, cos)
    Rt = np.einsum("nm,nmij->ij", -rbc * ms[None, :], sin)
    Zt = np.einsum("nm,nmij->ij", zbs * ms[None, :], cos)
    NR, NZ = -R * Zt, R * Rt  # (e_theta x e_zeta) in its R and Z components; the phi one is not representable
    N = np.sqrt(NR**2 + NZ**2 + (np.einsum("nm,nmij->ij", zbs * ms[None, :], cos)
                                 * np.einsum("nm,nmij->ij", rbc * ns[:, None] * nfp, sin)
                                 - Rt * np.einsum("nm,nmij->ij", -zbs * ns[:, None] * nfp, cos)) ** 2)
    nR, nZ = NR / N, NZ / N
    if direction is None:
        gR, gZ = nR / (nR**2 + nZ**2), nZ / (nR**2 + nZ**2)
    else:  # move along a given (R, Z) direction field, scaled so the normal displacement is h
        dR, dZ = direction(T, Zg)
        gR, gZ = dR / (nR * dR + nZ * dZ), dZ / (nR * dR + nZ * dZ)
    keep = ~((ms[None, :] == 0) & (ns[:, None] <= 0))
    keep_r = ~((ms[None, :] == 0) & (ns[:, None] < 0))
    A_r = cos[keep_r].reshape(int(keep_r.sum()), -1).T
    A_z = sin[keep].reshape(int(keep.sum()), -1).T
    pinv_r, pinv_z = np.linalg.pinv(A_r), np.linalg.pinv(A_z)
    m_top, n_top = min(int(max_mode), mpol - 1), min(int(max_mode), ntor)
    extra = [(m_top + 1, n) for n in range(-n_top, n_top + 1)] if extend else []
    modes = list(_dof_modes(inp, max_mode)) + extra + [(0, 0)]
    zero = replace(inp, rbc=np.zeros_like(rbc), zbs=np.zeros_like(zbs))
    columns = []
    for m, n in modes:
        h = np.cos(m * T - n * nfp * Zg)
        dr, dz = np.zeros_like(rbc), np.zeros_like(zbs)
        dr[keep_r], dz[keep] = pinv_r @ (h * gR).ravel(), pinv_z @ (h * gZ).ravel()
        columns.append(pack_boundary(replace(zero, rbc=dr, zbs=dz), max_mode, vary_major_radius=True))
    levels = np.asarray([max(abs(m), abs(n)) if (m, n) != (0, 0) else 1 for m, n in modes], dtype=float)
    return np.asarray(columns).T, modes, np.exp(-alpha * levels) / np.exp(-alpha)


def displacement_rms(h, modes) -> float:
    """RMS over the angles of ``sum h_mn cos(m theta - n nfp zeta)`` on ``modes`` (metres for metres).

    The normal displacement of the boundary for coefficients ``h`` on ``modes`` (``(m, n)`` pairs, as
    :attr:`ThreeTermsFreeBoundaryModel.boundary_modes`): every mode but ``(0, 0)`` has mean square ``h^2 / 2``.
    """
    h = np.asarray(h, dtype=float)
    weight = np.array([1.0 if (int(m), int(n)) == (0, 0) else 0.5 for m, n in modes])
    return float(np.sqrt(np.sum(weight * h**2)))


def newton_step_converged(step_rms: float, noise_rms: float, scale: float) -> bool:
    """Whether a Newton step of RMS size ``step_rms`` leaves nothing the rows can resolve: the free boundary's
    convergence test, shared by the Galerkin and least-squares solvers.

    Converged when ``step_rms <= max(noise_rms, sqrt(eps) * scale)``:

    - ``noise_rms`` is the step that the rows' measured noise alone produces (the difference of the rows of one
      boundary solved from two seeds, through the inverse Jacobian): a smaller step is not resolved by the rows;
    - ``sqrt(eps) * scale`` (``scale`` the coordinates' size, the major radius ``R0`` for a displacement) is
      round-off: Newton's error after a step ``p`` is ``O(|p|^2 / scale)``, so a step below ``sqrt(eps) scale``
      leaves an error at the coordinates' round-off ``eps * scale`` (the step tolerance of Dennis and Schnabel 1983, ch. 7).

    The step is in the coordinates' own units (metres of normal displacement: :func:`displacement_rms`), so the
    test does not depend on how the rows are scaled.
    """
    return bool(step_rms <= max(float(noise_rms), float(np.sqrt(np.finfo(float).eps)) * float(scale)))


def _newton_step(J, r):
    """The Newton step ``-J^-1 r`` of the square system, or ``None`` when ``J`` is numerically singular.

    Singular means no correct digit in the step: ``cond(J) * eps >= 1`` for ``J`` with its columns scaled to
    unit norm (the column scaling within ``sqrt(n)`` of the best one, van der Sluis 1969, so the coordinates'
    units do not decide it).
    """
    J, r = np.asarray(J, dtype=float), np.asarray(r, dtype=float)
    norms = np.linalg.norm(J, axis=0)
    if not np.all(np.isfinite(J)) or not np.all(norms > 0.0):
        return None
    if not np.linalg.cond(J / norms[None, :]) * np.finfo(float).eps < 1.0:
        return None
    return -np.linalg.solve(J, r)


#: Surface grid points per field period, in each direction, per harmonic of the deck's interior (``mpol``,
#: ``ntor``).  The rows integrate products of band-limited fields (edge field, geometry, test functions), which
#: needs more than 2 points per harmonic to be alias-free; 4 gives margin (the boundary then moves by far less
#: than its dependence on the interior ``ns``).  The solver's own residual report shares this grid; report final
#: numbers from an independent, converged evaluation.
_GRID_POINTS_PER_HARMONIC = 4

#: Iterations over which a trial solve's convergence rate is measured (:class:`~vmex.core.solver.EarlyStop`).
_TRIAL_RATE_WINDOW = 2000


def _trial_config(fixed, trial_ftol, device):
    """The implicit config of trial boundaries: final tolerance ``max(trial_ftol, FTOL)``, hot-restarted, and a solve
    that cannot reach the residual it is accepted at (``max_fsq_ratio * ftol``) within its budget fails early
    (:class:`~vmex.core.solver.EarlyStop`)."""
    from . import implicit as im

    ftols = np.array(fixed.ftol_array, dtype=float)
    ftols[-1] = max(float(trial_ftol), ftols[-1])
    loose = im.make_config(replace(fixed, ftol_array=ftols), multigrid=True, hot_restart=True)
    early_stop = EarlyStop(target=loose.max_fsq_ratio * loose.ftol, window=_TRIAL_RATE_WINDOW)
    return im._canonical_config(replace(loose, device=device, early_stop=early_stop))


def interface_grid(inp) -> tuple[int, int]:
    """``(nphi, ntheta)`` per field period of the interface rows of the deck ``inp``.

    ``_GRID_POINTS_PER_HARMONIC * max(mpol, ntor)`` in both directions (one rule for both: the edge field's
    harmonics follow the field lines, which mix the two).  The default of :class:`ThreeTermsFreeBoundaryModel`.
    """
    n = _GRID_POINTS_PER_HARMONIC * max(int(inp.mpol), int(inp.ntor), 1)
    return n, n


def interface_quadrature(nfp: int, nphi: int, ntheta: int) -> tuple[int, int]:
    """The virtual-casing singular quadrature ``(quad_nt, quad_np)`` for an ``nphi x ntheta`` field-period grid:
    ``(4 nfp nphi, 2 ntheta)``.  A coarser ``quad_np = ntheta`` leaves an error of order ``1e-4 |B|`` in the
    tangential plasma field; planning the quadrature from its own error estimate instead can exhaust GPU memory on
    a strongly shaped boundary.  The default of :class:`ThreeTermsFreeBoundaryModel`.
    """
    return 4 * int(nfp) * int(nphi), 2 * int(ntheta)


def solve_free_boundary_three_terms(
    inp: VmecInput,
    *,
    external_field: Any = None,
    mgrid_path=None,
    initial_boundary: VmecInput | None = None,
    max_mode: int | None = None,
    nphi: int | None = None,
    ntheta: int | None = None,
    digits: int = vc.VC_DIGITS,
    max_nfev: int = 60,
    quadrature: tuple[int, int] | None = None,
    trial_ftol: float | None = None,
    chunk: int = 8,
    bootstrap=None,
    bootstrap_helicity: int | None = None,
    previous=None,
    method: str = "galerkin",
    verbose: int = 0,
):
    """Free-boundary equilibrium of ``inp`` by the interface conditions of the module docstring.

    The free boundary is the deck's boundary displaced along the deck equilibrium's radial lines
    (:class:`ThreeTermsFreeBoundaryModel`); the deck fixes the labelling, so the answer does not depend on the start.
    Every trial is a fixed-boundary equilibrium on the deck's ``NS_ARRAY`` ladder at fixed toroidal flux, pressure
    and current (or iota).

    Parameters
    ----------
    inp:
        Free-boundary deck: ``PHIEDGE``, profiles, and the external field through ``external_field``,
        ``mgrid_path`` or ``MGRID_FILE``/``EXTCUR``.  The edge pressure must vanish; ``lasym`` must be false.
    external_field:
        Anything :func:`~vmex.core.virtual_casing.external_B_cartesian` accepts: an
        :class:`~vmex.core.mgrid.MgridField`, an ``xyz -> B`` callable (e.g. an ESSOS Biot-Savart field) or an
        object with ``b_cyl``.
    mgrid_path:
        An mgrid file, when ``external_field`` is not given.
    initial_boundary:
        Another surface to start from (only the starting displacement; default the deck's boundary).
    max_mode:
        The boundary's band (default :attr:`ThreeTermsFreeBoundaryModel.top_band`).
    nphi, ntheta:
        The rows' grid over one field period (default :func:`interface_grid`).
    digits, quadrature:
        Accuracy of the virtual-casing singular patches, planned once on the deck, and the singular quadrature
        ``(quad_nt, quad_np)`` (default :func:`interface_quadrature`).
    max_nfev:
        Most row evaluations of the Newton solve.
    trial_ftol, chunk:
        As for :class:`ThreeTermsFreeBoundaryModel`.
    bootstrap, bootstrap_helicity:
        Kinetic profiles and the quasi-symmetry helicity (0 for QA, +/-1 for QH; required with ``bootstrap``) to
        solve a Redl-self-consistent current with the boundary.  Redl's model is refused for QI fields.
    previous:
        An earlier result for the same deck (typically before a change of the external field): the solve starts
        from its boundary, Jacobian and model, and compiles nothing new.
    method:
        ``"galerkin"`` (the square system above, the faster) or ``"least_squares"``
        (:func:`~vmex.core.freeboundary_lsq.solve_free_boundary_least_squares`, the same keywords and result).
    verbose:
        Print progress when nonzero.

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``x`` (boundary coordinates, then any bootstrap-current values), ``fun`` (the rows), ``jac``, ``cost``,
        ``nfev``, ``njev``, ``success``; ``input`` (the free boundary, and the current, as a fixed-boundary deck);
        ``equilibrium`` (solved to ``_REPORT_FTOL``); ``model`` and ``aux`` (for ``previous=``); ``band`` and
        ``bands`` (each band solve's ``(max_mode, converged, seconds)``, :func:`solve_band_continuation`);
        ``boundary_residual`` and ``initial_boundary_residual`` (:class:`BoundaryResidual` at the end and at the
        start) and ``normal_outside_modes`` (:meth:`ThreeTermsFreeBoundaryModel.normal_outside_modes`).

    Warns
    -----
    RuntimeWarning
        When the solve does not lower ``B . n`` below the deck's, with the part outside the boundary's modes.
    """
    from scipy.optimize import OptimizeResult

    from .freeboundary import _external_field_from_input
    from .implicit import input_with_params
    from .optimize import solve_equilibrium

    if method == "least_squares":
        from .freeboundary_lsq import solve_free_boundary_least_squares

        return solve_free_boundary_least_squares(
            inp, external_field=external_field, mgrid_path=mgrid_path, initial_boundary=initial_boundary,
            max_mode=max_mode, nphi=nphi, ntheta=ntheta, digits=digits, max_nfev=max_nfev, quadrature=quadrature,
            trial_ftol=trial_ftol, chunk=chunk, bootstrap=bootstrap, bootstrap_helicity=bootstrap_helicity,
            previous=previous, verbose=verbose)
    if method != "galerkin":
        raise ValueError(f"method must be 'galerkin' or 'least_squares', got {method!r}")
    if bool(inp.lasym):
        raise NotImplementedError("the virtual-casing free boundary supports lasym = False only")
    _check_edge_pressure(inp)
    if external_field is None:
        external_field = _external_field_from_input(inp, mgrid_path)
    if previous is not None:
        model, x0, jacobian = previous.model, previous.x, previous.jac
        seed_state, seed_params = previous.aux[0], previous.aux[2]
    else:
        model = ThreeTermsFreeBoundaryModel(
            inp, max_mode=max_mode, nphi=nphi, ntheta=ntheta, digits=digits,
            quadrature=quadrature, trial_ftol=trial_ftol, chunk=chunk, bootstrap=bootstrap,
            bootstrap_helicity=bootstrap_helicity)
        x0 = None if initial_boundary is None else model.displacement_of(initial_boundary)
        jacobian = None
        seed_state, seed_params = model.seed[0], model.params0
    # The starting residuals are reports (and the reference of the warning below): their programs compile and run
    # on a background thread while the boundary is solved.  The warning's reference is the deck itself, also on a
    # warm restart (whose start is ``previous``'s answer).
    import threading

    def reports():
        initial = model.boundary_residual(seed_state, seed_params, external_field)
        deck = initial if previous is None else model.boundary_residual(model.seed[0], model.params0,
                                                                         external_field)
        return initial, deck

    report_box = []
    reporter = threading.Thread(target=lambda: report_box.append(_try(reports)), name="vmex-initial-residual",
                                daemon=True)
    reporter.start()
    model, out = solve_band_continuation(model, model.params0, external_field, x0=x0, jacobian=jacobian,
                                         max_nfev=max_nfev, verbose=verbose)
    reporter.join()
    if isinstance(report_box[0], BaseException):
        raise report_box[0]
    initial, deck = report_box[0]
    state, _, params, _ = out["aux"]
    boundary = input_with_params(model.fixed, params)
    # Restart from the rows' own state on its final rung (re-climbing the deck's ns ladder from an initial state
    # can stall short of the tolerance); an equilibrium that does not reach the tolerance is an error.
    equilibrium = solve_equilibrium(
        boundary, restart_from=state, forward_ftol=min(_REPORT_FTOL, float(np.asarray(boundary.ftol_array)[-1])),
        forward_max_iterations=max(int(np.asarray(boundary.niter_array)[-1]), 200000), raise_on_max_iterations=True)
    final = model.boundary_residual(equilibrium.solution, params, external_field)
    outside = model.normal_outside_modes(equilibrium.solution, params, external_field)
    if verbose:
        print(f"free boundary on band {model.max_mode}: {final}", flush=True)
    if final.normal >= deck.normal:
        import warnings

        warnings.warn(f"the free boundary did not lower B.n below the deck's ({final.normal:.2e} >= "
                      f"{deck.normal:.2e} RMS over |B|); {outside:.2e} of it lies outside the boundary's "
                      f"{len(model.boundary_modes)} modes, which the rows do not control", RuntimeWarning,
                      stacklevel=2)
    return OptimizeResult(
        x=out["x"], fun=out["rows"], jac=out["jacobian"], cost=0.5 * out["rows"] @ out["rows"], nfev=out["nfev"],
        njev=out["njev"], success=bool(out["converged"]), message=f"{out['accepted']} Newton steps",
        band=model.max_mode, bands=out.get("bands"),
        input=boundary, equilibrium=equilibrium, model=model, aux=out["aux"],
        boundary_residual=final, normal_outside_modes=outside, initial_boundary_residual=initial)


def solve_band_continuation(model, params, field, *, x0=None, jacobian=None, max_nfev=60, verbose=0):
    """The free boundary on ``model``'s band, by continuation in the band where Newton stalls: ``(model, fit)``.

    Newton with backtracking on ``model``'s band from ``x0`` (with ``jacobian``, of a nearby point, if given) comes
    first (:meth:`~ThreeTermsFreeBoundaryModel._newton_boundary`): where it converges this is that solve, at its
    cost.  Where it stalls -- a far start, or rows whose Newton basin is narrow on the top modes -- the band is
    built up from the lowest one (the poloidal modes ``m <= 1``): the band-1 boundary is Newton-solved from
    ``x0``'s surface, then the band is extended one mode at a time
    (:meth:`~ThreeTermsFreeBoundaryModel.with_band`), each band's Newton solve starting from the previous band's
    boundary (:meth:`~ThreeTermsFreeBoundaryModel.displacement_of`) and bootstrap current, up to ``model``'s band.
    Every band's Newton backtracks freely (not ``undamped``): the rows are ill-conditioned (condition 1e5 on
    finite-beta QA/QH at 12x12, a soft near-resonant (m, n) = (3, 1) boundary harmonic), so from a start inside the
    basin Newton may need shortened steps for several fresh Jacobians before full steps return (a 12x12 re-solve
    of an 8x8 QA root at 2.5 %: t = 1/64..1/4 for 8 fresh Jacobians, then quadratic to 1e-19); the two-shortened-
    steps rule declared such seeds without a free boundary.  Damped Newton still ends as a stall at a fresh step
    without descent (a minimum of ``|rows|^2`` that is no root), so nothing but a root is reported as converged.
    Optimizer trials keep the rule (:meth:`ThreeTermsFreeBoundaryModel.solve_boundary` with
    ``continuation=False``): a trial that fails fast only shortens the optimizer's step.
    Band 1, the start of the continuation, takes damped steps.  A band that does not converge ends the solve as a
    stall: ``converged=False``, with the last band that converged (``fit["band"]``, its model and fit; ``None`` and ``model``'s Newton result when none did).
    Each fit has ``band``, ``target`` (``model``'s band), ``bands`` (each solve's ``(max_mode, converged,
    seconds)``), ``direct`` (the Newton result on ``model``'s band) and the evaluation and Jacobian counts of
    the whole route.  Every band's rows are the square equations of the module docstring on that band, so a
    converged answer is the root of ``model``'s rows, whatever the route.
    """
    started = time.perf_counter()
    direct = model._newton_boundary(params, field, x0=x0, jacobian=jacobian, max_nfev=max_nfev, verbose=verbose)
    bands = [(model.max_mode, bool(direct["converged"]), time.perf_counter() - started)]
    target, nfev, njev = model.max_mode, direct["nfev"], direct["njev"]

    def result(answer, fit, band, converged):
        if verbose:
            print(f"band {band} of {target} ({'converged' if converged else 'stalled'}): " + ", ".join(
                f"{b} {'converged' if ok else 'stalled'} {seconds:.0f} s" for b, ok, seconds in bands), flush=True)
        return answer, dict(fit, converged=converged, nfev=nfev, njev=njev, band=band, target=target, bands=bands,
                            direct=direct)

    if direct["converged"]:
        return result(model, direct, target, True)
    if target <= 1:  # no lower band to start from
        return result(model, direct, None, False)
    reached, reached_fit = None, None  # the last band that converged
    current, start = model.with_band(1), None
    if x0 is not None:  # x0's surface on the lowest band's coordinates
        surface = model.with_boundary(model.params0, jnp.asarray(x0))
        start = current.displacement_of(SimpleNamespace(rbc=np.asarray(surface.rbc), zbs=np.asarray(surface.zbs)))
        start[current.n_boundary:] = np.asarray(x0)[model.n_boundary:]
    if verbose:
        print(f"band {target} stalled: continuation from band 1", flush=True)
    while True:
        started = time.perf_counter()
        try:
            fit = current._newton_boundary(params, field, x0=start, max_nfev=max_nfev, verbose=verbose)
        except (VmecError, RuntimeError):  # no certified equilibrium on the way: the band does not converge
            fit = dict(converged=False, nfev=0, njev=0)
        bands.append((current.max_mode, bool(fit["converged"]), time.perf_counter() - started))
        nfev, njev = nfev + fit["nfev"], njev + fit["njev"]
        if not fit["converged"]:
            break
        reached, reached_fit = current, fit
        if current.max_mode == target:
            return result(current, fit, target, True)
        wider = model if current.max_mode + 1 == target else current.with_band(current.max_mode + 1)
        boundary = fit["aux"][2]
        start = wider.displacement_of(SimpleNamespace(rbc=np.asarray(boundary.rbc), zbs=np.asarray(boundary.zbs)))
        start[wider.n_boundary:] = fit["x"][current.n_boundary:]
        if verbose:
            print(f"band {wider.max_mode}: from band {current.max_mode}'s boundary", flush=True)
        current = wider
    if reached is None:
        return result(model, direct, None, False)
    return result(reached, reached_fit, reached.max_mode, False)


def _radial_direction(state, resolution):
    """The edge's radial coordinate direction ``dX/ds`` as a function ``(theta, zeta) -> (dR, dZ)``.

    The boundary moves along the deck equilibrium's radial coordinate lines, not along the surface normal: VMEC's
    nested surfaces share their theta labelling along these lines, so the deck's labelling carries over to the free
    boundary without sliding.
    """
    xm, xn, rmnc, zmns = map(np.asarray, vc._state_geometry_spectra(state, resolution))
    ds = 1.0 / (rmnc.shape[0] - 1)
    dr = (rmnc[-1] - rmnc[-2]) / ds
    dz = (zmns[-1] - zmns[-2]) / ds

    def radial(T, Zg):
        a = xm[:, None, None] * T[None] - xn[:, None, None] * Zg[None]
        return np.tensordot(dr, np.cos(a), axes=1), np.tensordot(dz, np.sin(a), axes=1)

    return radial


def _surface_data(fixed, cfg, state, params, nphi, ntheta):
    """:func:`~vmex.core.virtual_casing.surface_field_data_from_state` of a concrete state as one compiled program.

    Eagerly, the field and surface assembly dispatch hundreds of small operations, each compiled on first use
    (seconds on a GPU before the first row); the program compiles once.  Its static fields (``nfp``, symmetry,
    ``signgs``, convention) are those of the traced construction.
    """
    from . import implicit as im

    static = {}

    def arrays(state, params):
        data = vc.surface_field_data_from_state(fixed, state, runtime=im.runtime_from_params(params, cfg),
                                                nphi=nphi, ntheta=ntheta)
        static.update(nfp=data.nfp, stellsym=data.stellsym, signgs=data.signgs,
                      source_convention=data.source_convention)
        return dict(gamma=data.gamma, B_total=data.B_total, normal=data.normal, area_vector=data.area_vector,
                    theta=data.theta, phi=data.phi)

    values = jax.jit(arrays)(state, params)
    return vc.VmecSurfaceFieldData(**values, **static)


def _certified(responses, report):
    """``responses`` if every column's linear solve met its certificate, else a ``RuntimeError``.

    The boundary Jacobian and the design gradients are built from these state responses; an uncertified one
    would enter them silently.  ``RuntimeError`` is what a trial's failed linearization raises: the Newton
    solve stops, and the single-stage problem rejects the trial.
    """
    converged = np.asarray(report.converged)
    if not bool(np.all(converged)):
        worst = float(np.max(np.asarray(report.residual_norm) / np.asarray(report.tolerance)))
        raise RuntimeError(f"{int(np.sum(~converged))} of {converged.size} state responses of the boundary "
                           f"Jacobian missed their certificate (worst residual {worst:.3g} x the tolerance)")
    return responses


def _try(function, *args):
    """``function(*args)``, or the exception it raised (for a worker thread to hand back)."""
    try:
        return function(*args)
    except BaseException as error:  # noqa: BLE001 -- re-raised by the caller
        return error


class ThreeTermsFreeBoundaryModel:
    """The free boundary of one deck for many external fields and plasma parameters.

    The fixed-boundary equilibria and their derivatives come from :mod:`vmex.core.implicit`, with the boundary,
    PHIEDGE and current profile as traced :class:`~vmex.core.implicit.ImplicitParams`; the external field enters
    the compiled rows as an argument, so neither a new field nor new plasma parameters recompile anything.  The
    rows are the square equations of the module docstring, with the quadrature plan and the field scale ``B0``
    fixed on the deck.

    Parameters
    ----------
    inp:
        The deck.  Its pressure must vanish at the boundary (the rows carry no sheet current).
    max_mode:
        The boundary's band (default :attr:`top_band`, the interior's ``max(mpol - 1, ntor)``).
    nphi, ntheta:
        The rows' grid over one field period (default :func:`interface_grid`).
    digits:
        Accuracy the virtual-casing singular patches are planned to.
    chunk:
        Most Jacobian columns per batch.
    device:
        JAX device of the equilibria (default the first).
    trial_ftol:
        A looser force tolerance for trial boundaries; a state is re-solved to the deck's before it is
        differentiated.  A trial solve that cannot reach the residual it is accepted at within the deck's
        ``NITER`` stops early (:class:`~vmex.core.solver.EarlyStop`).
    quadrature:
        The virtual-casing singular quadrature ``(quad_nt, quad_np)`` (default :func:`interface_quadrature`;
        ``quad_nt`` a multiple of ``nfp * nphi``).
    bootstrap, bootstrap_helicity:
        :class:`~vmex.core.bootstrap.KineticProfiles` and the quasi-symmetry helicity (0 for QA, +/-1 for QH;
        non-quasi-symmetric fields are refused): the current profile becomes Redl's bootstrap current in the same
        solve.  ``x`` then gains the current's knot values (one on the axis, then one per half-mesh surface) after
        the ``n_boundary`` boundary coordinates, and the rows gain one self-consistency row per knot
        (:class:`~vmex.core.bootstrap.HalfMeshCurrent`).
    fixed_boundary:
        With ``bootstrap``: hold the boundary and solve only the current (no external field, no virtual casing).

    Attributes
    ----------
    x0, x_scale:
        The deck's coordinates (zero displacement) and their scales.  ``x`` is the displacement of the deck's
        boundary on the band's modes (:attr:`boundary_modes`), then any current values.
    params0, seed:
        The deck's ``ImplicitParams`` and its equilibrium ``(state, mask)``.
    with_boundary:
        ``with_boundary(params, x)``: ``params`` with the boundary (and current) of ``x``.
    rows:
        ``rows(state, params, field)``, the traceable rows (``interface_rows`` their interface part).

    Methods: :meth:`solve_boundary` (the free boundary, cold or warm), :meth:`evaluate` (rows at a boundary),
    :meth:`linearize` (their Jacobian and state responses), :meth:`push` and :meth:`pullback` (forward and reverse
    derivatives of other functions of the equilibrium), :meth:`with_band`, :meth:`rereferenced`,
    :meth:`displacement_of`, :meth:`boundary_residual`, :meth:`bootstrap_residual` and :meth:`solve`.
    """

    def __init__(self, inp: VmecInput, *, max_mode=None, nphi=None, ntheta=None, digits=vc.VC_DIGITS, chunk=8,
                 device=None,
                 trial_ftol=None, quadrature=None, bootstrap=None, bootstrap_helicity=None,
                 fixed_boundary=False):
        from . import implicit as im
        from .bootstrap import HalfMeshCurrent

        self._im = im
        # the deck and options this model was built from (rereferenced: the same model on another deck)
        self._source = (inp, dict(nphi=nphi, ntheta=ntheta, digits=digits, chunk=chunk, device=device,
                                  trial_ftol=trial_ftol, quadrature=quadrature, bootstrap=bootstrap,
                                  bootstrap_helicity=bootstrap_helicity, fixed_boundary=fixed_boundary))
        self.fixed = fixed = replace(inp, lfreeb=False, mgrid_file="NONE")
        self.bootstrap = None if bootstrap is None else HalfMeshCurrent(fixed, bootstrap, bootstrap_helicity)
        self.fixed_boundary = bool(fixed_boundary)
        self.deck = None  # the finite-beta rows' deck response data (_interface_setup)

        # Vacuum: B_out is the coil field alone, no virtual casing (_is_vacuum).
        self.vacuum = bootstrap is None and _is_vacuum(fixed)
        if self.fixed_boundary and self.bootstrap is None:
            raise ValueError("fixed_boundary=True solves only the bootstrap current: pass bootstrap=")
        if not self.fixed_boundary:  # the interface rows assume p(1) = 0, as the public solve does
            _check_edge_pressure(fixed)
        if self.bootstrap is not None:
            self.fixed = fixed = self.bootstrap.deck
        # The band of boundary modes: ``max_mode``, by default the interior's (``top_band``).
        self.top_band = int(max(fixed.mpol - 1, fixed.ntor))
        self.max_mode = int(self.top_band if max_mode is None else max_mode)
        device = jax.devices()[0] if device is None else device
        # Hot-restarted solves are exact: the m=1 angle gauge is fixed (implicit.ImplicitConfig.freeze_m1), so
        # the state is a single-valued function of the boundary whatever the seed, and the implicit Jacobian is
        # the derivative of the re-solve map.
        # Certified means refined to |F| <= refine_tol: the rows and their implicit derivatives are taken at the
        # refined state, which must be the root (a refinement that stops short is a failed evaluation).
        cfg = im.make_config(fixed, multigrid=True, hot_restart=True, certify_refined=True)
        self.cfg = cfg = im._canonical_config(replace(cfg, device=device))
        # Trial points (the boundary steps) may converge to a looser force residual: their rows move by far less
        # than the interface floor, and a state is solved to the deck's tolerance before it is differentiated.
        self.cfg_trial = cfg if trial_ftol is None else _trial_config(fixed, trial_ftol, device)
        if trial_ftol is not None:
            im._template_runtime(self.cfg_trial)
        self.params0 = im.params_from_input(fixed, device=device)
        im._template_runtime(cfg)
        self.R0 = float(np.asarray(fixed.rbc)[int(fixed.ntor), 0])  # the length scale of the convergence test
        self.chunk = int(chunk)
        self._warm_programs()
        seed = self.solve(self.params0)
        if seed is None:
            raise VmecConvergenceError("the initial boundary has no converged fixed-boundary equilibrium")
        state, mask = seed[:2]
        self.seed = (state, mask)
        # The seed boundary's field scale B0 (area-RMS |B|) scales the interface rows, and with its mean poloidal
        # circumference the bootstrap rows (HalfMeshCurrent.set_unit): every row is a field over B0.
        grid = interface_grid(fixed)
        self.nphi, self.ntheta = int(nphi or grid[0]), int(ntheta or grid[1])
        surface = _surface_data(fixed, cfg, state, self.params0, self.nphi, self.ntheta)
        area = np.linalg.norm(np.asarray(surface.area_vector), axis=0)
        self.B0 = float(np.sqrt(np.sum(area * np.sum(np.asarray(surface.B_total) ** 2, axis=0)) / np.sum(area)))
        if self.bootstrap is not None:
            gamma = np.asarray(surface.gamma)  # (3, nphi, ntheta) over one field period
            length = float(np.mean(np.sum(np.linalg.norm(np.roll(gamma, -1, axis=2) - gamma, axis=0), axis=1)))
            self.bootstrap.set_unit(length, self.B0)
        self._radial = None
        self.precision = None
        if not self.fixed_boundary:
            self._radial = _radial_direction(state, self.cfg.resolution)
            self.quadrature = tuple(map(int, quadrature or interface_quadrature(fixed.nfp, self.nphi, self.ntheta)))
            self.digits = digits
            # the virtual-casing plan of the deck's surface, for every band (_set_band)
            if not self.vacuum:
                self.precision = vc.plan_vc_precision(surface, digits=digits, quad_nt=self.quadrature[0],
                                                      quad_np=self.quadrature[1])
        self._set_band(self.max_mode)

    def _set_band(self, max_mode):
        """The boundary coordinates on the modes up to ``max_mode`` and the rows of that band.

        Everything else -- the deck's equilibrium, field scale, grid and virtual-casing plan -- belongs to the
        deck and is shared by every band (:meth:`with_band`).
        """
        from .optimize import boundary_arrays_from_x, pack_boundary

        fixed = self.fixed
        self.max_mode = int(max_mode)
        # Boundary coordinates: the normal displacement h of the deck's surface (see _normal_displacement_map):
        # one function for the one normal condition, no tangential (relabelling) directions among the unknowns.
        x_ref = pack_boundary(fixed, self.max_mode, vary_major_radius=True)
        if self.fixed_boundary:
            self.x0, self.x_scale, normal_map = x_ref[:0], x_ref[:0], None
        else:
            normal_map, self.boundary_modes, self.x_scale = _normal_displacement_map(
                fixed, self.max_mode, direction=self._radial, extend=self.vacuum)
            self.x0 = np.zeros(normal_map.shape[1])
        self.n_boundary = nb = self.x0.size
        if self.bootstrap is not None:
            self.x0 = np.r_[self.x0, self.bootstrap.x0]
        self._linearization = None
        self._last_linearize = None  # (key, extra key, result, _linearization) of the latest linearize
        self._x_ref = jnp.asarray(x_ref)
        self._normal_map = None if normal_map is None else jnp.asarray(normal_map)

        def with_boundary(params, x):
            if not self.fixed_boundary:
                rbc, zbs = boundary_arrays_from_x(fixed, self._x_ref + self._normal_map @ x[:nb], self.max_mode,
                                                  vary_major_radius=True)
                params = replace(params, rbc=rbc, zbs=zbs)
            return params if self.bootstrap is None else self.bootstrap.apply(params, x[nb:])

        self.with_boundary = with_boundary
        if self.bootstrap is not None:
            self.x_scale = np.r_[self.x_scale, self.bootstrap.x_scale]
        self._implicit_setup()
        if not self.fixed_boundary:
            self._interface_setup(self.seed[0], self.nphi, self.ntheta, self.digits)

    def with_band(self, max_mode):
        """The same deck's model on the boundary modes up to ``max_mode``, without solving the deck again."""
        import copy

        other = copy.copy(self)
        other.deck, other._warm = None, None
        other.__dict__.pop("_row_noise", None)  # the rows' measured noise is that of a band's own rows
        other._set_band(max_mode)
        return other

    def rereferenced(self, x):
        """This model with the boundary at coordinates ``x`` as its deck, on the same band, grid and options.

        The coordinates belong to the deck: a boundary far from it is described along the deck's radial lines and
        in the deck's theta labelling, and Newton from the deck may not reach it.  The new model's ``x0`` (zero
        displacement) is that boundary, with the bootstrap current values of ``x``.
        """
        if self.fixed_boundary:
            raise ValueError("a fixed-boundary model has no boundary to re-reference")
        inp, options = self._source
        boundary = self.with_boundary(self.params0, jnp.asarray(x))
        rbc, zbs = np.asarray(boundary.rbc), np.asarray(boundary.zbs)  # the deck's arrays, the band's modes set
        other = ThreeTermsFreeBoundaryModel(replace(inp, rbc=rbc, zbs=zbs), max_mode=self.max_mode, **options)
        nb = self.n_boundary
        other.x0 = np.r_[other.x0[:nb], np.asarray(x, dtype=float)[nb:]]
        return other

    def _warm_programs(self):
        """Compile the solve's largest programs on background threads while the seed equilibrium is solved.

        The seed solve's Newton-refinement programs (:func:`~vmex.core.implicit.warm_refinement`) and the first
        linearization's tangent programs (:func:`~vmex.core.implicit.warm_staged_tangent`) depend only on the
        deck's structure, so they compile in parallel with the multigrid ladder (XLA compilation releases the
        GIL).  :meth:`linearize` joins the tangent thread before its first tangent solve.  Cache warming only:
        the same executables.
        """
        import threading

        self._warm = None
        im, cfg, params, chunk = self._im, self.cfg, self.params0, self.chunk
        if not im._overlapping_compiles_pay():
            return

        def refinement():
            try:
                im.warm_refinement(cfg, params)
            except Exception:  # cache warming only: the real calls compile on demand
                pass

        def tangents():
            try:
                im.warm_staged_tangent(cfg, params, chunk=chunk, certify_rtol=cfg.adjoint_tol)
            except Exception:  # cache warming only
                pass

        threading.Thread(target=refinement, name="vmex-refine-warm", daemon=True).start()
        self._warm = threading.Thread(target=tangents, name="vmex-tangent-warm", daemon=True)
        self._warm.start()

    def _interface_setup(self, state, nphi, ntheta, digits):
        """The interface rows, with the quadrature plan of the seed ``state`` (:attr:`precision`)."""
        from .freeboundary import _vacuum_scalars

        im, cfg, fixed = self._im, self.cfg, self.fixed
        nfp = int(fixed.nfp)
        precision = self.precision  # None in vacuum: the coils' field alone, no virtual-casing quadrature
        if self.vacuum:
            digits = None
        # A closed curve inside the coils for G_coil: the seed's magnetic axis (its wout raxis_cc / zaxis_cs, read
        # from the solved seed state; the circulation is the same on any loop that links the coils).
        _, _, rmnc, zmns = map(np.asarray, vc._state_geometry_spectra(state, cfg.resolution))
        n_axis = int(cfg.resolution.ntor) + 1
        axis = _axis_loop(SimpleNamespace(raxis_cc=rmnc[0, :n_axis], zaxis_cs=zmns[0, :n_axis], nfp=nfp))
        vacuum, B0 = self.vacuum, self.B0
        # Surface data are (nphi, ntheta) over one field period.
        phi, theta = np.meshgrid(np.linspace(0.0, 2.0 * np.pi / nfp, nphi, endpoint=False),
                                 np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False), indexing="ij")
        # The boundary arrays' entries as series in (m, n nfp).
        ntor_b, mpol_b = np.shape(self.params0.rbc)[0] // 2, np.shape(self.params0.rbc)[1]
        modes = np.meshgrid(np.arange(-ntor_b, ntor_b + 1) * nfp, np.arange(mpol_b), indexing="ij")
        xm, xn = jnp.asarray(modes[1].ravel(), dtype=float), jnp.asarray(modes[0].ravel(), dtype=float)
        # R0 d/dh: B.grad = (B/R)(d_phi + iota d_theta), so R0 times the h-gradient of the mean square of
        # B.n/B0 is a dimensionless moment of B.n/B0 (as are the finite-beta rows, with the same factor).
        length = float(np.asarray(fixed.rbc)[int(fixed.ntor), 0])
        phi_grid = jnp.asarray(phi[:, 0])
        if vacuum:
            # The boundary arrays are linear in x: their derivatives along x.
            boundary = jax.jacfwd(lambda x: self.with_boundary(self.params0, x))(jnp.asarray(self.x0))
            to_x = (boundary.rbc.reshape(-1, self.x0.size), boundary.zbs.reshape(-1, self.x0.size))

        def interface_rows(state, params, field):
            """The interface rows of the module docstring at ``state`` (solved on ``params``' boundary)."""
            deck, field = (field.deck, field.field) if isinstance(field, _DeckField) else (None, field)
            runtime = im.runtime_from_params(params, cfg)
            G_coil = jnp.abs(_net_current(field, *axis))
            G = jnp.abs(_vacuum_scalars(state, runtime)[1])
            current = ((G - G_coil) / G_coil)[None]
            if vacuum:  # no plasma current: the field outside is the coils' own, an explicit function of h
                gradient = jax.grad(lambda r, z: _mean_square_normal_field(
                    r.ravel(), z.ravel(), xm, xn, phi, theta, phi_grid, field), argnums=(0, 1))(params.rbc, params.zbs)
                rows = to_x[0].T @ gradient[0].ravel() + to_x[1].T @ gradient[1].ravel()
                return jnp.concatenate([length * rows[:-1] / (2.0 * B0**2), current])
            data = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
            area = jnp.linalg.norm(jnp.asarray(data.area_vector), axis=0)
            weight = (area / jnp.sum(area)).ravel()
            B_out = vc.PlasmaVacuumInterface.from_surface_data(
                data, p_edge=0.0, digits=digits, precision=precision).total_B_out(field)
            Bn = (jnp.sum(B_out * jnp.asarray(data.normal), axis=0) / B0).ravel()
            # test functions: the exact response of the pointwise sqrt(w) B.n / |B_in| to each boundary coordinate
            # at the deck (plasma response included; _deck_response), for this field: R0 d r_deck / d h_k
            if deck is None:
                raise ValueError("the finite-beta rows need the deck's response: pass model.attach(field)")
            (X0, n0, s0, Bp0), (dX, dn, ds, dBp) = deck
            Bc0 = vc.external_B_cartesian(field, X0, phi_grid)
            unit = jnp.eye(3)[:, :, None, None] * jnp.ones_like(X0)[None]
            DBc = jax.vmap(lambda e: jax.jvp(lambda X: vc.external_B_cartesian(field, X, phi_grid), (X0,), (e,))[1])(unit)
            base = Bc0 + Bp0
            moved = jnp.einsum("iopq,kipq->kopq", DBc, dX) + dBp  # the field's change at the moving deck points
            tests = (ds * jnp.sum(base * n0, axis=0)[None]
                     + s0[None] * (jnp.sum(moved * n0[None], axis=1) + jnp.sum(base[None] * dn, axis=1)))
            B_in = jnp.asarray(data.B_total)
            r = (jnp.sqrt(weight) * Bn * B0 / jnp.linalg.norm(B_in, axis=0).ravel())
            return jnp.concatenate([length * tests[:-1].reshape(tests.shape[0] - 1, -1) @ r, current])

        self.interface_rows = interface_rows
        if not vacuum:
            self.deck = self._deck_response(nphi, ntheta, digits, precision)
        self._normal_field = self._make_normal_field(fixed, nphi, ntheta, digits, precision)
        self._interface = lambda state, runtime, field: _interface_terms(  # traceable, from a runtime
            fixed, state, field, runtime=runtime, nphi=nphi, ntheta=ntheta, digits=digits, precision=precision,
            p_edge=0.0)[:2]
        self._terms = jax.jit(lambda state, params, field: self._interface(
            state, im.runtime_from_params(params, cfg), field))

    def _make_normal_field(self, fixed, nphi, ntheta, digits, precision):
        """The traceable ``(B_out . n / |B_in|, area weights)`` of a state on the ``(nphi, ntheta)`` grid, as the
        interface rows evaluate it: the coils' field in vacuum, plus the plasma's by virtual casing at finite beta
        (:meth:`normal_outside`; both methods' interface setups install it)."""
        vacuum = self.vacuum

        def normal_field(state, runtime, field):
            field = field.field if isinstance(field, _DeckField) else field
            data = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
            area = jnp.linalg.norm(jnp.asarray(data.area_vector), axis=0)
            if vacuum:
                B_out = vc.external_B_cartesian(field, jnp.asarray(data.gamma), jnp.asarray(data.phi))
            else:
                B_out = vc.PlasmaVacuumInterface.from_surface_data(
                    data, p_edge=0.0, digits=digits, precision=precision).total_B_out(field)
            normal = jnp.sum(B_out * jnp.asarray(data.normal), axis=0) / jnp.linalg.norm(jnp.asarray(data.B_total), axis=0)
            return normal, area / jnp.sum(area)

        return normal_field

    def attach(self, field, deck=None):
        """``field`` with the deck's response data (``deck``, default :attr:`deck`) for the compiled rows.

        The identity in vacuum (no deck data) and on a field that already carries them.
        """
        deck = self.deck if deck is None else deck
        return field if deck is None or isinstance(field, _DeckField) else _DeckField(field, deck)

    def _deck_response(self, nphi, ntheta, digits, precision):
        """The finite-beta rows' test space: the deck's boundary data and their exact responses to ``h``.

        They depend on the deck alone (its equilibrium and boundary coordinates, the grid, the virtual-casing
        precision), not on the coils, so they are kept on disk next to the persistent compilation cache
        (:func:`_deck_cache_load`): a model built again on the same deck (a restart, a second start) reads them
        instead of repeating the tangent solve and the virtual-casing push.  The key hashes every input, the seed state's bytes and the vmex sources,
        so a changed deck, resolution or code never reuses an entry.

        The pointwise residual ``r = sqrt(w) B_out . n / |B_in|`` (``w`` the area weights) at the deck depends on
        the boundary coordinates through the deck equilibrium (points, normal, weights, ``|B_in|`` and the
        plasma's own field ``B_plasma``, all independent of the coils) and through the coil field at the moving
        points.  Returned: those deck data ``(X, n, s, B_plasma)`` (``s = sqrt(w) / |B_in|``) and their implicit
        tangents along each boundary coordinate (one block-factorized tangent solve at the seed state); the rows
        combine them with the coil field and its gradient at the deck points.
        """
        im, cfg, fixed, nb = self._im, self.cfg, self.fixed, self.n_boundary
        state, mask = self.seed
        directions = jax.tree.map(lambda a: a[:nb], self.boundary_directions(self.params0, self.x0))
        if self._warm is not None:  # its programs are this call's: wait instead of compiling them twice
            self._warm.join()
            self._warm = None

        def geometry(state, params):
            runtime = im.runtime_from_params(params, cfg)
            data = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
            interface = vc.PlasmaVacuumInterface.from_surface_data(data, p_edge=0.0, digits=digits,
                                                                    precision=precision)
            area = jnp.linalg.norm(jnp.asarray(data.area_vector), axis=0)
            s = jnp.sqrt(area / jnp.sum(area)) / jnp.linalg.norm(jnp.asarray(data.B_total), axis=0)
            return jnp.asarray(data.gamma), jnp.asarray(data.normal), s, interface.B_plasma

        key = _digest("deck-response", fixed, nphi, ntheta, digits, precision, self.max_mode, self.boundary_modes,
                      self._x_ref, self._normal_map, state, mask, directions, _source_digest(), jax.__version__,
                      jax.devices()[0].device_kind)

        def compute():
            dz = self._tangents(self.params0, state, mask, directions)
            primal = jax.jit(geometry)(state, self.params0)
            tangent = jax.jit(lambda st, mk, prm, d, b: self.push(geometry, st, mk, prm, d, b))(
                state, mask, self.params0, dz, directions)
            return primal, tangent

        return _deck_cached(key, compute)

    def _implicit_setup(self):
        """The rows (interface, then bootstrap) and their implicit tangents, pushes and pullbacks."""
        im, cfg = self._im, self.cfg

        def rows(state, params, field):
            parts = [] if self.fixed_boundary else [self.interface_rows(state, params, field)]
            if self.bootstrap is not None:
                parts.append(self.bootstrap.rows(state, im.runtime_from_params(params, cfg), params))
            return jnp.concatenate(parts)

        self.rows = rows
        self._rows = jax.jit(rows)
        active = im._active_state_fields(cfg)
        edge = im._edge_mask(cfg)

        def tangents(params, state, mask, batch, column_scales=None):
            """Projected state responses to the stacked parameter tangents: one block factorization, chunked.

            Staged programs (:func:`~vmex.core.implicit.staged_tangent_multi_rhs`): the factors from the
            forward refinement's compiled program, then the block solve and the GMRES corrector.  A column the
            corrector and the refinement leave uncertified is an error (:func:`_certified`), never a silent
            column of the Jacobian.  The columns are certified to the adjoint tolerance (``adjoint_tol``), not the
            looser one for least-squares Jacobians: they are the boundary Jacobian of a square Newton solve and
            the implicit-function-theorem design gradients, which must be exact.  The block-direct responses
            usually meet it outright (the raw block system is the exact radial linearization).  ``column_scales``:
            the columns' units, in which they are certified against the largest right-hand side
            (:func:`~vmex.core.implicit.staged_tangent_multi_rhs`)."""
            return _certified(*im.staged_tangent_multi_rhs(
                params, cfg, state, mask, batch, chunk=self.chunk,
                certify_rtol=cfg.adjoint_tol, certify_maxiter=cfg.adjoint_maxiter, column_scales=column_scales))

        def push(fun, state, mask, params, dz_batch, params_batch):
            """Columns of ``fun(state, params)`` along the stacked responses, one at a time.

            As upstream's implicit Jacobian: the state of a column is assembled from its projected response and
            its parameters inside the map, so no batch of full states is ever built.  The columns are mapped
            in small batches, not vectorized over the chunk: a column's linearized rows hold the interface field on
            every surface point at once (a coil Biot-Savart sum over all segments, the virtual-casing quadrature),
            so a vectorized chunk multiplies the peak memory for no speed.
            """
            frozen = jax.lax.stop_gradient(state)
            P = im._dof_projector(cfg, mask)

            def at(z, prm):
                return fun(im._assemble(z, im.runtime_from_params(prm, cfg), frozen, P, edge), prm)

            # Two columns per step (vmex.core.loops): wider batches are a little faster at several times the peak
            # memory, and a single column is a program XLA:GPU miscompiles.
            return map_columns(lambda a: jax.jvp(at, (P(frozen), params), (P(a[0]), a[1]))[1],
                               (dz_batch, params_batch), batch_size=2)

        def predict(state, mask, dz_batch, dx, params):
            """First-order trial state: the anchor plus the responses times ``dx``, on the trial boundary."""
            frozen = jax.lax.stop_gradient(state)
            P = im._dof_projector(cfg, mask)
            z = jax.tree.map(lambda a, d: a + jnp.tensordot(dx, d, axes=1), P(frozen), dz_batch)
            return im._assemble(z, im.runtime_from_params(params, cfg), frozen, P, edge)

        self._tangents = tangents
        self.push = push
        push_rows = jax.jit(lambda state, mask, params, dz, pb, field: push(
            lambda s, p: rows(s, p, field), state, mask, params, dz, pb))

        def push_rows_padded(state, mask, params, dz, pb, field):
            """``push_rows`` one whole chunk at a time: one executable for any number of columns."""
            n = jax.tree.leaves(dz)[0].shape[0]
            state, mask, params, dz, pb = im._canonical_arrays(cfg, (state, mask, params, im._padded(dz, self.chunk),
                                                                     im._padded(pb, self.chunk)))
            pieces = [push_rows(state, mask, params, jax.tree.map(lambda a, lo=lo: a[lo:lo + self.chunk], dz),
                                jax.tree.map(lambda a, lo=lo: a[lo:lo + self.chunk], pb), field)
                      for lo in range(0, jax.tree.leaves(dz)[0].shape[0], self.chunk)]
            return jnp.concatenate(pieces)[:n]

        self._push_rows = push_rows_padded
        # First-order trial state from the last linearization (upstream's perturbation warm start): it keeps
        # the trial on the path the Jacobian was taken along, instead of a hot restart that may drift.
        self._predict = jax.jit(predict)

        def pullback(fun, state, mask, params, cotangents, *args, factors=None):
            """``ImplicitParams`` gradients of ``cotangents[i] . fun(state, params, *args)`` through the equilibrium.

            Reverse mode with one block factorization shared by every row (upstream's implicit pullback): for a few
            scalar rows of an expensive function this costs a few reverse passes, not one forward pass per
            parameter direction.  ``fun`` may be a tuple of functions with ``cotangents`` a tuple of their rows'
            cotangents: each row then runs only its own function's reverse pass, and the rows come out in order.
            """
            frozen = jax.lax.stop_gradient(state)
            back = im._block_state_pullback(params, cfg, frozen, mask, active_fields=active,
                                            probe_chunk_size=self.chunk, factors=factors)
            funs, blocks = (fun, cotangents) if isinstance(fun, tuple) else ((fun,), (cotangents,))
            parts = []
            for f, c in zip(funs, blocks):
                _, vjp = jax.vjp(lambda s, p, f=f: f(s, p, *args), frozen, params)

                def row(c, vjp=vjp):
                    state_bar, params_bar = vjp(c)
                    return jax.tree.map(jnp.add, params_bar, back(state_bar)[0])

                parts.append(map_columns(row, c))  # a vjp per row: see vmex.core.loops
            return jax.tree.map(lambda *a: jnp.concatenate(a), *parts)

        self.pullback = pullback

        def adjoint(state, mask, params, state_bars, params_bars, factors=None):
            """Rows' ``ImplicitParams`` gradients from their state and direct parameter cotangents.

            :meth:`pullback` in two programs: the rows' cotangents ``(state_bars, params_bars)`` of each function's
            own reverse pass (stacked, any number of functions), then this program, the block adjoint through the
            equilibrium shared by every row, ``params_bar + back(state_bar)`` per row.  The same arithmetic as
            :meth:`pullback`, whose single program holds one copy of the adjoint per function.
            """
            frozen = jax.lax.stop_gradient(state)
            back = im._block_state_pullback(params, cfg, frozen, mask, active_fields=active,
                                            probe_chunk_size=self.chunk, factors=factors)
            return map_columns(lambda row: jax.tree.map(jnp.add, row[1], back(row[0])[0]), (state_bars, params_bars))

        self.adjoint = adjoint

    def displacement_of(self, boundary, ntheta=96, nzeta=48):
        """Coordinates ``x`` (normal displacement of the reference surface) closest to ``boundary``'s surface.

        Each reference point is moved along its normal to the nearest point of the other surface (nearest-point
        iteration on a dense sampling), and the displacements are fitted on the normal modes.
        """
        from .optimize import boundary_arrays_from_x

        nfp = int(self.fixed.nfp)

        def points(rbc, zbs, nt, nz, full=False):
            rbc, zbs = np.asarray(rbc, dtype=float), np.asarray(zbs, dtype=float)
            ntor = (rbc.shape[0] - 1) // 2
            ms, ns = np.arange(rbc.shape[1]), np.arange(-ntor, ntor + 1)
            th = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
            ze = np.linspace(0.0, 2.0 * np.pi / (1 if full else nfp), nz, endpoint=False)
            T, P = np.meshgrid(th, ze, indexing="ij")
            ph = ms[None, :, None, None] * T[None, None] - ns[:, None, None, None] * nfp * P[None, None]
            R = np.einsum("nm,nmij->ij", rbc, np.cos(ph)); Z = np.einsum("nm,nmij->ij", zbs, np.sin(ph))
            Rt = np.einsum("nm,nmij->ij", -rbc * ms, np.sin(ph)); Zt = np.einsum("nm,nmij->ij", zbs * ms, np.cos(ph))
            Rp = np.einsum("nm,nmij->ij", rbc * ns[:, None] * nfp, np.sin(ph))
            Zp = np.einsum("nm,nmij->ij", -zbs * ns[:, None] * nfp, np.cos(ph))
            X = np.stack([R * np.cos(P), R * np.sin(P), Z], -1)
            et = np.stack([Rt * np.cos(P), Rt * np.sin(P), Zt], -1)
            ep = np.stack([Rp * np.cos(P) - R * np.sin(P), Rp * np.sin(P) + R * np.cos(P), Zp], -1)
            N = np.cross(et, ep)
            return T, P, X, N / np.linalg.norm(N, axis=-1, keepdims=True)

        from scipy.spatial import cKDTree
        r0, z0 = boundary_arrays_from_x(self.fixed, self._x_ref, self.max_mode, vary_major_radius=True)
        T, P, X, n = points(r0, z0, ntheta, nzeta)
        _, _, Y, _ = points(boundary.rbc, boundary.zbs, 4 * ntheta, 4 * nzeta * nfp, full=True)
        tree = cKDTree(Y.reshape(-1, 3))
        t = np.zeros(T.shape)
        for _ in range(4):  # the point of the other surface on the normal line, by nearest-point iteration
            _, k = tree.query((X + t[..., None] * n).reshape(-1, 3))
            t = np.sum((Y.reshape(-1, 3)[k].reshape(X.shape) - X) * n, axis=-1)
        basis = np.stack([np.cos(m * T - k * nfp * P).ravel() for m, k in self.boundary_modes], axis=1)
        h = np.linalg.lstsq(basis, t.ravel(), rcond=None)[0]
        return np.r_[h, np.zeros(self.n_boundary - h.size), self.x0[self.n_boundary:]]

    def solve(self, params, seed=None, tight=True):
        """Hot-restarted fixed-boundary equilibrium at ``params``: ``(state, mask)``, or ``None`` if not certified.

        ``tight=False`` solves to the trial tolerance, without the Newton refinement that certifies a state for
        differentiation (:meth:`linearize` re-solves tight first).
        """
        im, cfg = self._im, (self.cfg if tight else self.cfg_trial)
        if seed is not None:
            im._PERTURB_SEED[cfg] = seed
        params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), params)
        if tight:
            state, mask, status, _, _ = im._host_solve_and_mask_status(cfg, params_np)
            if int(status) != 0:
                return None
        else:  # a trial point is not differentiated: the forward solve alone, without the Newton refinement
            with im._device_context(cfg):
                try:
                    state, mask = im._host_solve_and_mask_impl(cfg, params_np, refine=False)
                except (VmecError, RuntimeError):  # RuntimeError: the relayed sentinel of a VmecError
                    return None
            result = im._LAST_SOLVE[cfg][1]
            fsq = float(result.fsqr) + float(result.fsqz) + float(result.fsql)
            if not (bool(result.converged) or fsq / cfg.ftol <= cfg.max_fsq_ratio):
                return None
        return jax.device_put(jax.tree.map(jnp.asarray, (state, mask)), cfg.device)

    def boundary_residual(self, state, params, field) -> BoundaryResidual:
        """:class:`BoundaryResidual` of a solved state."""
        return summarize_boundary_residual(*self._terms(state, params, field))

    def normal_outside_modes(self, state, params, field) -> float:
        """RMS (over ``|B|``, area-weighted) of the part of ``B_out . n`` outside the boundary's modes.

        The rows control ``B . n`` through the boundary's modes; what lies outside them (higher poloidal or
        toroidal harmonics) only resolution lowers.  Harmonics of the grid's FFT, in field periods.
        """
        rows, weights = self._terms(state, params, field)
        return float(self._outside_rms(jnp.asarray(rows)[0], jnp.asarray(weights)))

    def normal_outside(self, state, runtime, field):
        """Traceable :meth:`normal_outside_modes` of a state and its runtime: design-differentiable, the value of
        the :data:`NORMAL_OUTSIDE_MODES` quantity of :class:`ThreeTermsFreeBoundaryProblem`.

        Only ``B_out . n / |B_in|`` is evaluated, as the interface rows do (the coils' field in vacuum, plus the
        plasma's by virtual casing at finite beta); the pressure-balance and sheet-current terms of
        :meth:`boundary_residual` would add their own field evaluations (56 GiB in one program at 48 x 48)."""
        return self._outside_rms(*self._normal_field(state, runtime, field))

    def _outside_rms(self, normal, weights):
        f = normal * jnp.sqrt(weights * weights.size)  # mean of f^2 = the weighted mean square of B.n
        power = jnp.abs(jnp.fft.fft2(f) / f.size) ** 2  # axes (phi, theta)
        k_phi = np.fft.fftfreq(f.shape[0], 1.0 / f.shape[0]).astype(int)
        k_theta = np.fft.fftfreq(f.shape[1], 1.0 / f.shape[1]).astype(int)
        inside = np.zeros(f.shape, dtype=bool)
        for m, n in self.boundary_modes:  # sin(m theta - n nfp phi) holds (-n, m) and (n, -m)
            inside |= (k_phi[:, None] == -n) & (k_theta[None, :] == m)
            inside |= (k_phi[:, None] == n) & (k_theta[None, :] == -m)
        return jnp.sqrt(jnp.sum(jnp.where(inside, 0.0, power)))

    def bootstrap_residual(self, state, params) -> float:
        """``max_j |I'(s_j) - I'_Redl(s_j)| / max_j |I'_Redl(s_j)|`` of a solved state (with ``bootstrap``)."""
        difference, target = map(np.asarray, self.bootstrap.mismatch(
            state, self._im.runtime_from_params(params, self.cfg), params))
        return float(np.max(np.abs(difference)) / np.max(np.abs(target)))

    def evaluate(self, x, params, field, seed=None, tight=False):
        """Rows at boundary ``x``, or ``None`` if the solve failed.

        Returns ``(rows, (state, mask, params_x, tight))``; ``tight`` marks a state solved to the deck's
        tolerance rather than the trial one.
        """
        params_x = self.with_boundary(params, jnp.asarray(x))
        if seed is None and self._linearization is not None:
            x_ref, state_ref, mask_ref, dz = self._linearization
            seed = self._predict(state_ref, mask_ref, dz, jnp.asarray(np.asarray(x) - x_ref), params_x)
        tight = tight or self.cfg_trial is self.cfg
        solved = self.solve(params_x, seed, tight)
        if solved is None:
            return None
        state, mask = solved
        return (np.asarray(self._rows(state, params_x, self.attach(field))),
                (state, mask, params_x, tight))

    def boundary_directions(self, params, x):
        """``ImplicitParams`` tangents of the boundary coordinates, stacked."""
        eye = jnp.eye(np.size(x))
        return jax.vmap(lambda e: jax.jvp(lambda xx: self.with_boundary(params, xx), (jnp.asarray(x),), (e,))[1])(eye)

    def linearize(self, x, aux, field, extra=None, extra_scale=None):
        """Columns of the rows along the model coordinates and the ``extra`` parameter tangents.

        The model coordinates are the boundary, then the bootstrap current.  A trial state is first solved to
        the deck's tolerance (from itself).  Returns ``(J, dz, params_batch, aux)``: the Jacobian, the projected
        state responses and parameter tangents of its columns, and the (tight) state they belong to.  The
        responses also seed the next trials (first-order predicted states).  A call at the boundary, plasma
        parameters and field of the latest one returns its result (without ``extra``'s columns when not asked
        for): a polish that stalls asks again for the Jacobian it was given.
        """
        state, mask, params_x, tight = aux
        key = _fingerprint(x, params_x, field)
        extra_key = None if extra is None else _fingerprint(extra)
        last = self._last_linearize
        if last is not None and last[0] == key and extra_key in (None, last[1]):
            J, dz, batch, aux = last[2]
            self._linearization = last[3]
            if extra_key is None and last[1] is not None:  # drop the extra columns
                n = self.x0.size
                J = J[:, :n]
                dz, batch = jax.tree.map(lambda a: a[:n], dz), jax.tree.map(lambda a: a[:n], batch)
            return J, dz, batch, aux
        if not tight:
            solved = self.solve(params_x, seed=state)
            if solved is None:
                raise RuntimeError("the trial state does not converge to the deck's tolerance")
            state, mask = solved
            aux = (state, mask, params_x, True)
        directions = self.boundary_directions(params_x, x)
        nb = self.n_boundary
        # Boundary, bootstrap-current and extra directions each in a program of its own: one batch of every
        # direction can exhaust the GPU.
        groups = [jax.tree.map(lambda a: a[:nb], directions)] if nb else []
        if self.bootstrap is not None:
            groups.append(jax.tree.map(lambda a: a[nb:], directions))
        n_coordinates = len(groups)
        if extra is not None:
            groups.append(extra)
        join = lambda *a: jnp.concatenate(a)  # noqa: E731
        if self._warm is not None:  # its programs are this call's: wait instead of compiling them twice
            self._warm.join()
            self._warm = None
        # One tangent solve (one block factorization) for every group; the pushes stay per group.
        # every column in the units the Newton solve and the design use (x_scale; extra_scale for the extra
        # directions): the tangent certificate then holds them to one absolute accuracy in those units
        n_extra = 0 if extra is None else jax.tree.leaves(extra)[0].shape[0]
        units = np.r_[np.asarray(self.x_scale, dtype=float),
                      np.ones(n_extra) if extra_scale is None else np.asarray(extra_scale, dtype=float)]
        dz_all = self._tangents(params_x, state, mask, jax.tree.map(join, *groups), units)
        ends = np.cumsum([jax.tree.leaves(batch)[0].shape[0] for batch in groups])
        parts = [(jax.tree.map(lambda a: a[lo:hi], dz_all), batch)
                 for lo, hi, batch in zip(np.r_[0, ends[:-1]], ends, groups)]
        columns = np.hstack([np.asarray(self._push_rows(state, mask, params_x, dz, batch, self.attach(field))).T
                             for dz, batch in parts])
        dz = jax.tree.map(join, *[dz for dz, _ in parts[:n_coordinates]])
        self._linearization = (np.asarray(x, dtype=float).copy(), state, mask, dz)
        if extra is not None:
            dz = jax.tree.map(join, dz, parts[-1][0])
        batch = jax.tree.map(join, *[batch for _, batch in parts])
        self._last_linearize = (key, extra_key, (columns, dz, batch, aux), self._linearization)
        return columns, dz, batch, aux

    def rows_at(self, aux, field):
        """The rows of a solved ``aux = (state, mask, params, tight)`` (no solve)."""
        state, _, params_x, _ = aux
        return np.asarray(self._rows(state, params_x, self.attach(field)))

    def repeat(self, x, params, field, aux):
        """The rows at ``x`` solved again, to the deck's tolerance, from a second seed; ``None`` if that fails.

        The difference from the rows of ``aux`` (the same boundary, solved from another start) is the rows'
        repeatability, the noise below which Newton cannot converge (:func:`newton_step_converged`).  The second
        seed is the deck's equilibrium, or a cold start when ``x`` is the deck's boundary itself; the solver's
        one-entry memos are cleared so that the boundary is really solved again.
        """
        im, cfg = self._im, self.cfg
        params_x = self.with_boundary(params, jnp.asarray(x))
        at_deck = not np.any(np.asarray(x)[:self.n_boundary] - self.x0[:self.n_boundary])
        for memo in (im._LAST_SOLVE, im._LAST_REFINED, im._LAST_REFINEMENT_CORRECTION) + (
                (im._HOT_CACHE,) if at_deck else ()):
            memo.pop(cfg, None)
        solved = self.solve(params_x, None if at_deck else self.seed[0], tight=True)
        return None if solved is None else self.rows_at((*solved, params_x, True), field)

    def _step_converged(self, step, noise_step, x):
        """:func:`newton_step_converged` for the boundary part of ``step`` (metres, against ``R0``) and for the
        bootstrap-current part (against the current values ``x``); ``noise_step`` is the step of the measured
        row noise, or ``None`` before it is measured."""
        nb = self.n_boundary
        noise = np.zeros_like(step) if noise_step is None else noise_step
        ok = nb == 0 or newton_step_converged(displacement_rms(step[:nb], self.boundary_modes),
                                              displacement_rms(noise[:nb], self.boundary_modes), self.R0)
        if step.size > nb:  # the bootstrap current: plain RMS of its knot values
            rms = lambda v: float(np.sqrt(np.mean(np.square(v))))  # noqa: E731
            ok = ok and newton_step_converged(rms(step[nb:]), rms(noise[nb:]), rms(x[nb:]))
        return ok

    def _newton_boundary(self, params, field, *, x0=None, jacobian=None, max_nfev=60, verbose=0, undamped=False):
        """The boundary where the square rows vanish: Newton steps with backtracking.

        Jacobian: a given ``jacobian`` (of a nearby point) is reused while each step reduces the rows ten-fold.  At
        finite beta, where a fresh Jacobian costs many row evaluations, a first slower step updates it by Broyden's
        rank-1 secant correction; a second slow step, or a failed one, refreshes it at the current point.

        Convergence: the Newton step at a tight evaluation is within :func:`newton_step_converged`'s tolerance --
        round-off, or the step the rows' measured noise produces.  The noise is measured once a fresh Newton step
        finds no descent (:meth:`_measure_noise`).  A singular Jacobian, or a fresh step without descent that is
        larger than the noise's, is a stall (``converged=False``): a minimum of ``|rows|^2`` that is no root is never
        reported as converged.  Trial points solved to ``trial_ftol`` are re-solved to the deck's tolerance before
        convergence is tested.

        ``undamped=True`` stops, not converged and with ``damped=True``, when two fresh-Jacobian steps in a row
        descend only once shortened: a refreshed Jacobian that does not restore full steps means the point is
        outside the rows' Newton basin.  (One shortened step is common on the way in from a far start.)
        """
        x = np.asarray(self.x0 if x0 is None else x0, dtype=float)
        got = self.evaluate(x, params, field)
        for fraction in (0.75, 0.5, 0.25, 0.0):  # a far start may have no equilibrium: back off toward the deck
            if got is not None or not np.any(x):  # (this only moves the start, never the answer)
                break
            x = fraction * x
            got = self.evaluate(x, params, field)
        if got is None:
            raise RuntimeError("the starting boundary has no certified equilibrium")
        r, aux = got
        J, fresh, slow, nfev, njev, steps, started = jacobian, False, False, 1, 0, 0, time.perf_counter()
        # the rows' measured noise (a vector of row differences), kept by the model for later solves of its rows
        noises = self.__dict__.setdefault("_row_noise", {})
        converged, tight, noise = False, self.cfg_trial is self.cfg, noises.get("rows")
        damped, updated = False, False  # updated: the kept Jacobian carries a Broyden update
        shortened = False  # the last fresh-Jacobian step descended only once shortened
        while nfev < max_nfev:
            if J is None:
                J, _, _, aux = self.linearize(x, aux, field)
                J, fresh, njev, updated = np.asarray(J), True, njev + 1, False
                if not got[1][3] and aux[3]:  # the rows of the tight state the Jacobian belongs to
                    r, got = self.rows_at(aux, field), (None, aux)
            p = _newton_step(J, r)
            if p is None:  # singular Jacobian: no Newton step
                if not fresh:
                    J = None
                    continue
                if verbose:
                    print("Newton stalled: singular Jacobian", flush=True)
                break
            # the step of a Jacobian at this point or a nearby one (the previous point, ``previous=``, the
            # single-stage anchor) is the distance to the root to first order
            if self._step_converged(p, None if noise is None else _newton_step(J, noise), x):
                if got[1][3]:  # converged: the last step leaves an error O(|p|^2), kept if it does not raise the rows
                    converged, final = True, self.evaluate(x + p, params, field, tight=True)
                    nfev += 1
                    if final is not None and final[0] @ final[0] <= r @ r:
                        x, (r, aux), steps = x + p, final, steps + 1
                    break
                tight, got = True, self.evaluate(x, params, field, tight=True)  # converge on tight rows only
                nfev += 1
                if got is None:
                    break
                r, aux = got
                continue
            if slow:  # the last step was slow: test convergence with its Jacobian, then refresh it
                J, slow = None, False
                continue
            cost = 0.5 * float(r @ r)
            t, accepted = 1.0, None
            while t >= (1.0 / 64 if fresh else 1.0) and nfev < max_nfev:  # a reused J: its full step, or refresh
                trial = self.evaluate(x + t * p, params, field, tight=tight)
                nfev += 1
                if trial is not None and 0.5 * float(trial[0] @ trial[0]) < (1.0 - 1e-4 * t) * cost:
                    accepted = trial
                    break
                t *= 0.25
            if verbose:
                new = np.inf if accepted is None else 0.5 * float(accepted[0] @ accepted[0])
                print(f"Newton step {steps + 1}{' (fresh J)' if fresh else ''}: t {t:.3g}, cost {cost:.6e} -> "
                      f"{new:.6e} ({time.perf_counter() - started:.1f} s)", flush=True)
            if fresh and accepted is not None:
                # a refreshed Jacobian that again descends only along a shortened step: outside the Newton basin
                if undamped and t < 1.0 and shortened:  # keep the shortened step's progress
                    x, (r, aux), steps, damped = x + t * p, accepted, steps + 1, True
                    break
                shortened = t < 1.0
            if accepted is None:
                if fresh and not tight:  # the trials' looser force tolerance is the rows' noise: solve them tight
                    tight, J, got = True, None, self.evaluate(x, params, field, tight=True)
                    nfev += 1
                    if got is None:  # no certified equilibrium at the deck's tolerance
                        break
                    r, aux = got
                    continue
                if fresh:  # no descent along a fresh Newton step: the rows' noise, or a stall
                    if got[1][3]:
                        noise, converged, used = self._measure_noise(x, p, r, J, aux, params, field, noise,
                                                                     max_nfev - nfev)
                        nfev += used
                        if noise is not None:
                            noises["rows"] = noise
                    if not converged and verbose:
                        print(f"Newton stalled: cost {cost:.6e}, step not within the rows' noise", flush=True)
                    break
                J = None
                continue
            new_cost = 0.5 * float(accepted[0] @ accepted[0])
            s, r_new = t * p, np.asarray(accepted[0])
            x, got, steps = x + s, accepted, steps + 1
            slow = new_cost > 1e-2 * cost  # slower than a 10x residual reduction a step (a chord method)
            if slow and not updated and not self.vacuum:  # (vacuum: a Jacobian costs about one evaluation)
                # Broyden's update (Broyden 1965) instead of a fresh Jacobian: the rank-1 correction that makes
                # J reproduce the step's observed change of the rows (J_new s = r_new - r).  A Jacobian costs as
                # much as many row evaluations at finite beta, so one secant step is tried first; a second slow
                # step, or a failed one, refreshes the Jacobian as before.
                J = J + np.outer(r_new - r - J @ s, s) / float(s @ s)
                slow, updated = False, True
            else:
                updated = False
            r, aux = got
            fresh = False  # the Jacobian (if kept) is now of the previous point
        return dict(x=x, rows=np.asarray(r), jacobian=J, aux=aux, nfev=nfev, njev=njev, accepted=steps,
                    converged=converged, damped=damped, noise=None if noise is None else float(np.linalg.norm(noise)))

    def _measure_noise(self, x, p, r, J, aux, params, field, noise, budget):
        """The rows' noise at a tight point ``x`` where the fresh Newton step ``p`` finds no descent.

        Two measurements, each kept when larger than the noise so far: the rows of the same boundary solved again
        from a second seed (:meth:`repeat`), and, for noise that is reproducible but not smooth in the boundary,
        the rows' deviation from the Jacobian's linear model along ``p`` at round-off length (where the model's
        second-order error is below round-off).  Converged when ``p`` is within the noise's step, or the rows are
        within twice the noise (a difference of two draws).  Returns ``(noise, converged, evaluations)``; at most
        ``budget`` evaluations.
        """
        converged, used = False, 0
        if used < budget:
            again = self.repeat(x, params, field, aux)
            used += 1
            if again is not None:
                noise = again - r if noise is None or (np.linalg.norm(again - r)
                                                       > np.linalg.norm(noise)) else noise
                converged = self._step_converged(p, _newton_step(J, noise), x)
        if not converged and used < budget:
            nb = self.n_boundary
            length = displacement_rms(p[:nb], self.boundary_modes) if nb else 0.0
            q = p * (np.sqrt(np.finfo(float).eps) * self.R0 / length) if length > 0.0 else 0.0 * p
            trial = self.evaluate(x + q, params, field, tight=True)
            used += 1
            if trial is not None:
                deviation = np.asarray(trial[0]) - (r + J @ q)
                if noise is None or np.linalg.norm(deviation) > np.linalg.norm(noise):
                    noise = deviation
                converged = (self._step_converged(p, _newton_step(J, noise), x)
                             or np.linalg.norm(r) <= 2.0 * np.linalg.norm(noise))
        return noise, converged, used

    def solve_boundary(self, params, field, *, x0=None, jacobian=None, max_nfev=60, verbose=0, continuation=True):
        """The free boundary for ``params`` and ``field`` on this model's band, from ``x0`` (default: the reference
        surface): :func:`solve_band_continuation`, Newton with backtracking (:meth:`_newton_boundary`) and, where it
        stalls, the band built up from ``m <= 1``.  ``continuation=False`` is Newton alone (``undamped``): a stall
        is returned at once, without the band continuation (an optimizer's trial, whose design has another start).

        Returns a dict with ``x``, ``rows``, ``jacobian``, ``aux`` (state, mask, params, tight), ``nfev``,
        ``njev``, ``accepted`` (steps), ``converged``, ``band`` (the last band that converged, ``None`` if none)
        and ``bands``; a stall (``converged=False``) returns this band's Newton result.
        """
        if not continuation:
            return self._newton_boundary(params, field, x0=x0, jacobian=jacobian, max_nfev=max_nfev, verbose=verbose,
                                         undamped=True)
        model, fit = solve_band_continuation(self, params, field, x0=x0, jacobian=jacobian, max_nfev=max_nfev,
                                             verbose=verbose)
        if model is self:
            return fit
        return dict(fit["direct"], converged=False, nfev=fit["nfev"], njev=fit["njev"], band=fit["band"],
                    target=fit["target"], bands=fit["bands"])


#: Force residual of the reported equilibrium's final rung: plain descent stopped at the usual 1e-13 leaves an
#: error of order 1 % in a quasi-symmetry error near 1e-6 (the boundary solve's own states are Newton-refined).
_REPORT_FTOL = 1e-17


@dataclass(frozen=True)
class _AcceptedPoint:
    """The design the optimizer last accepted (``ThreeTermsFreeBoundaryProblem.accepted``)."""

    parameters: np.ndarray


class _NormalOutsideModes:
    """The :data:`NORMAL_OUTSIDE_MODES` quantity (evaluated by the problem with its current model)."""

    __name__ = "normal_outside_modes"

    def __call__(self, *args):
        raise TypeError("NORMAL_OUTSIDE_MODES is evaluated by ThreeTermsFreeBoundaryProblem, not called")


#: A ``parameter_quantities`` entry of :meth:`ThreeTermsFreeBoundaryProblem.from_loss`: the RMS of ``B_out . n /
#: |B_in|`` outside the boundary's modes (:meth:`ThreeTermsFreeBoundaryModel.normal_outside`), the part of the
#: normal field the truncated boundary cannot cancel. The square rows zero ``B . n`` only on the boundary's modes,
#: so an optimizer sees nothing of this part; bounding it keeps designs whose free boundary a finer boundary
#: resolves (an unbounded run can drift to coils with no free boundary at higher resolution).
NORMAL_OUTSIDE_MODES = _NormalOutsideModes()


class ThreeTermsFreeBoundaryProblem(FunctionProblem):
    """A scalar loss and constraint rows of a design ``x`` at its three-terms free boundary.

    Build with ``FreeBoundaryProblem.from_loss(..., boundary_condition="three_terms")``
    (:meth:`~vmex.core.freeboundary_problem.FreeBoundaryProblem.from_loss`); pass it to
    :func:`vmex.core.optimize.minimize` (``method="SLSQP"`` or ``"AUGLAG"``).  ``loss``, ``quantities``, ``parameter_quantities``,
    ``field_from_parameters`` and ``plasma_from_parameters`` are as there.  Every trial is the K = 0 boundary
    of the field and plasma parameters of ``x`` (:class:`ThreeTermsFreeBoundaryModel`), warm-started from the
    first-order prediction of the latest linearized solution; a trial without a certified solution, or whose
    boundary fit did not converge from any start (a stalled Newton solve is no free boundary, whatever its
    residual), raises :class:`~vmex.core.errors.TrialRejected`.  A row ``h`` (loss or quantity) has the design
    gradient of the implicit function theorem of the (square) boundary equations,

        dh/dx = dh/dx|_b - lambda^T dr/dx,   lambda = J_b^-T dh/db,

    with ``r`` the interface rows, ``J_b`` their Jacobian in the model's coordinates ``b`` (boundary, and the
    bootstrap current with ``bootstrap``) and ``dr/dx`` through the plasma parameters and the field; every
    row's ``dh/db`` and plasma term come from one reverse pass through the equilibrium.  Nothing recompiles
    when ``x`` changes.

    A trial that no start reaches on the current deck is solved on a new deck, the latest linearized boundary
    (:meth:`ThreeTermsFreeBoundaryModel.rereferenced`), which is kept when the trial converges there.

    Attributes: ``model`` (the current :class:`ThreeTermsFreeBoundaryModel`), ``references`` (the step, trial and
    deck of every deck change, the first being the original deck), ``accepted`` and ``accepted_step`` (the latest
    accepted design), and ``stats`` (counts of trials, failures, boundary evaluations, linearizations and
    re-references).
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_max_nfev=100, boundary_start=None,
                  **model_options):
        """Solve the seed design's free boundary.

        ``model_options`` are :class:`ThreeTermsFreeBoundaryModel` keywords (``nphi``, ``ntheta``,
        ``quadrature``, ``chunk``, ``trial_ftol``, ``bootstrap``, ...).  A trial's boundary is Newton-solved from
        the accepted point's first-order prediction with its Jacobian (refreshed when it no longer converges
        fast), to at most ``boundary_max_nfev`` evaluations; a fit that does not converge within them is no free
        boundary: the trial tries its next start, or fails.

        ``boundary_start`` is where the seed's solve starts instead of the deck's own boundary: the deck model's
        coordinates themselves (an array of ``x0``'s shape, e.g. a saved step's :meth:`boundary_coordinates` on the
        same deck: that step's root), or anything with ``rbc`` and ``zbs`` (a saved step's LCFS), fitted on the deck
        by :meth:`ThreeTermsFreeBoundaryModel.displacement_of`.  The deck still defines the problem: a restart on the run's deck from its step's boundary
        starts near that step's root (the fit measures along the normal, the coordinates along the deck's radial
        lines) and solves the run's very equations.
        """
        functions = (loss, field_from_parameters, *quantities, *parameter_quantities)
        if plasma_from_parameters is not None:
            functions += (plasma_from_parameters,)
        if not all(map(callable, functions)):
            raise TypeError("loss, parameter maps and quantities must be callable")
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary optimization requires JAX_ENABLE_X64=1")
        model = ThreeTermsFreeBoundaryModel(inp, **model_options)
        seed_fit = None
        if not model.fixed_boundary:
            # the seed design's free boundary, by band continuation where the direct solve stalls
            # (solve_band_continuation); a stall below the band leaves the seed without a free boundary
            x = jnp.asarray(x0, dtype=float)
            params = model.params0 if plasma_from_parameters is None else plasma_from_parameters(model.params0, x)
            if boundary_start is None:
                start = None
            elif isinstance(boundary_start, np.ndarray) and boundary_start.shape == np.shape(model.x0):
                start = np.array(boundary_start, dtype=float)
            else:
                start = model.displacement_of(boundary_start)
            solved, seed_fit = solve_band_continuation(model, params, field_from_parameters(x), x0=start,
                                                       max_nfev=boundary_max_nfev)
            if not seed_fit["converged"]:
                raise VmecError(f"the seed design has no three-terms free boundary on band {model.max_mode} (the "
                                f"band continuation stalled after band {seed_fit['band']})")
            model = solved
        return cls(model, loss, x0, field_from_parameters=field_from_parameters,
                   plasma_from_parameters=plasma_from_parameters, scales=scales, names=names,
                   quantities=tuple(quantities), parameter_quantities=tuple(parameter_quantities),
                   boundary_max_nfev=boundary_max_nfev, seed_fit=seed_fit)

    def __init__(self, model, loss, x0, *, field_from_parameters, plasma_from_parameters, scales, names, quantities,
                 parameter_quantities, boundary_max_nfev, seed_fit=None):
        self.field = field_from_parameters
        self.boundary_max_nfev = int(boundary_max_nfev)
        x0 = np.asarray(x0, dtype=float)
        self._functions = (loss, tuple(quantities), tuple(parameter_quantities), plasma_from_parameters)
        self._bind(model, x0)
        self._cache, self._anchor = {}, {}
        self.references = [dict(step=0, trial=0, deck=model.fixed)]  # the deck of each step on (_rereference)
        self._warm_threads = self._warm_gradient_programs(x0)
        self.stats = dict(trials=0, failed=0, boundary_evaluations=0, linearizations=0)
        if seed_fit is not None and seed_fit["converged"]:  # the seed's free boundary, already solved
            self.stats["trials"] += 1
            self._store(x0, seed_fit)
        if self._solve(x0) is None:
            raise VmecError("the seed design has no three-terms free boundary")
        self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0
        super().__init__(x0, names=names, scales=scales, fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})

    def _bind(self, model, x0):
        """The programs of the rows and their design gradients on ``model`` (and its deck's coordinates)."""
        from .implicit import runtime_from_params

        loss, quantities, parameter_quantities, plasma_from_parameters = self._functions
        field_from_parameters = self.field
        self.model = model
        params_at = (lambda params, x: params) if plasma_from_parameters is None else plasma_from_parameters

        def seed_params_at(x):
            return params_at(model.params0, x)

        self._params_at = jax.jit(seed_params_at)

        def part(f, with_x):
            if isinstance(f, _NormalOutsideModes):  # needs the model (its modes and interface): this one's
                def outside(state, params, x):
                    # traced with checkpointed quadrature scans: its reverse pass (the design gradient) would
                    # otherwise keep every source chunk's kernel intermediates (62 GiB at 48 x 48); values unchanged
                    with vc.remat_scans():
                        return jnp.atleast_1d(model.normal_outside(state, runtime_from_params(params, model.cfg),
                                                                   field_from_parameters(x)))
                outside.__name__ = "normal_outside_modes"
                return outside

            def value(state, params, x):
                runtime = runtime_from_params(params, model.cfg)
                return jnp.ravel(jnp.asarray(f(state, runtime, x) if with_x else f(state, runtime)))
            value.__name__ = getattr(f, "__name__", "row")  # names its programs in compile logs
            return value

        parts = (part(loss, True), *(part(f, False) for f in quantities),
                 *(part(f, True) for f in parameter_quantities))

        def rows(state, params, x):
            return jnp.concatenate([f(state, params, x) for f in parts])

        self._rows = jax.jit(rows)
        # The coordinates of x that move plasma parameters: their columns of the interface rows enter the gradient.
        tangent = jax.jacfwd(seed_params_at)(jnp.asarray(x0))
        moved = sum(np.any(np.asarray(a).reshape(-1, x0.size) != 0, axis=0) for a in jax.tree.leaves(tangent))
        self._plasma = np.flatnonzero(moved)
        eye = jnp.eye(x0.size)[self._plasma]

        def mapped_params(x):
            """The plasma map of the deck's parameters at ``x``, on fixed model coordinates.

            The map is always applied to the deck's parameters (as :meth:`_solve` does), never to a trial's own
            mapped ones: a map that depends on its input (a multiplicative pressure scale) would otherwise have its
            derivative at a moved point scaled by the map itself.  The model coordinates overwrite the boundary
            (and bootstrap current), so the derivative in ``x`` does not depend on which coordinates are used."""
            return model.with_boundary(params_at(model.params0, x), jnp.asarray(model.x0))

        self._plasma_directions = jax.jit(lambda x: jax.vmap(lambda e: jax.jvp(mapped_params, (x,), (e,))[1])(eye))

        def cotangents(f):
            """The program of ``f``'s rows' cotangents ``(state_bar, params_bar)``: its own reverse pass only."""
            def program(state, params, x):
                frozen = jax.lax.stop_gradient(state)
                _, vjp = jax.vjp(lambda s, p: f(s, p, x), frozen, params)
                return map_columns(vjp, jnp.eye(jax.eval_shape(f, state, params, x).size))  # see vmex.core.loops
            program.__name__ = f"cotangents_{f.__name__}"
            return jax.jit(program)

        # One small program per function and one for the block adjoint they share (model.adjoint), not one program
        # with a copy of the adjoint per function: they compile in parallel (_warm_gradient_programs), and the
        # adjoint's executable depends only on the config and the row count.
        self._parts = parts  # (state, params, x) -> rows, one function per row block, in row order
        self._cotangents = tuple(cotangents(f) for f in parts)
        self._adjoint = jax.jit(lambda state, mask, params, state_bars, params_bars, factors: model.adjoint(
            state, mask, params, state_bars, params_bars, factors=factors))

        @jax.jit
        def coordinates(params, b, x, params_bar):
            """``ImplicitParams`` cotangents onto the model coordinates ``b`` and onto ``x`` (plasma parameters)."""
            _, b_vjp = jax.vjp(lambda z: model.with_boundary(params, z), b)
            _, x_vjp = jax.vjp(mapped_params, x)
            return jax.vmap(lambda bar: (b_vjp(bar)[0], x_vjp(bar)[0]))(params_bar)

        self._coordinates = coordinates
        # the direct x-Jacobian part by part: one reverse pass vmapped over all rows would replicate every part's
        # backward program (a virtual-casing quantity's, 0.85 GiB) once per row of every other part (56 GiB)
        def direct(state, params, x):
            return jnp.concatenate([jax.jacrev(f, argnums=2)(state, params, x).reshape(-1, x.size) for f in parts])

        self._direct = jax.jit(direct)
        # the deck's response data are an argument (model.attach), not constants of these programs
        self._field_vjp = jax.jit(lambda state, params, x, lam, deck: jax.vjp(
            lambda z: model.rows(state, params, model.attach(field_from_parameters(z), deck)), x)[1](lam)[0])
        self._field_jvp = jax.jit(lambda state, params, x, dx, deck: jax.jvp(
            lambda z: model.rows(state, params, model.attach(field_from_parameters(z), deck)), (x,), (dx,))[1])

    def _solve(self, x):
        """The trial at ``x`` (cached): boundary fit, state and rows, or ``None``."""
        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        if key in self._cache:
            return self._cache[key]
        self.stats["trials"] += 1
        params, field = self._params_at(jnp.asarray(x)), self.field(jnp.asarray(x))
        # Warm starts first, each with the anchor's Jacobian: its first-order boundary prediction, then its own
        # boundary.  They are near the answer by construction, so Newton from them must not need damping; one that
        # does is outside the Newton basin (a large design step) and hands over at once, instead of crawling with
        # damped steps.  The last start is the cold solve from the deck: the same answer by any route.
        starts = [(None, None)]
        anchor = self._anchor
        if anchor:
            dx = x - anchor["x"]
            dr = anchor["J_p"] @ dx[self._plasma] + np.asarray(self._field_jvp(
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx), self.model.deck))
            starts = [(anchor["b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0], anchor["J_b"]),
                      (anchor["b"], anchor["J_b"]), (None, None)]
        fit = None
        for start, J in starts:
            try:
                if J is None:  # cold: Newton from the deck (a trial that stalls there fails: the optimizer shortens
                    # its step; the band continuation, tens of minutes at finite beta, is the seed's route only)
                    fit = self.model.solve_boundary(params, field, x0=start, max_nfev=self.boundary_max_nfev,
                                                    continuation=False)
                else:
                    fit = self.model._newton_boundary(params, field, x0=start, jacobian=J,
                                                      max_nfev=self.boundary_max_nfev, undamped=True)
            except (VmecError, RuntimeError):  # an uncertified equilibrium: try the next start
                continue
            if not fit["converged"]:  # a stalled fit is no free boundary (its rows are not zero): the next start
                self.stats["boundary_evaluations"] += fit["nfev"]
                fit = None
                continue
            break
        if fit is None and anchor:
            fit = self._rereference(x, anchor)
        if fit is None:
            self.stats["failed"] += 1
            self._cache[key] = None
            return None
        return self._store(x, fit)

    _BOUND = ("model", "_params_at", "_rows", "_plasma", "_plasma_directions", "_parts", "_cotangents",
              "_adjoint", "_coordinates", "_direct", "_field_vjp", "_field_jvp")

    def _rereference(self, x, anchor):
        """The trial at ``x`` on a new deck, the anchor's boundary, when the current deck reaches it from no start.

        The deck fixes the coordinates (radial lines, theta labelling) and the cold start; a design that has
        moved far from it -- coils several cm away after tens of steps -- may have a free boundary that Newton
        from the deck no longer reaches although it exists.  The anchor's boundary, the latest linearized
        solution, is near it by construction.  The new deck is kept when the trial converges on it (the anchor
        and the uncertified trials, whose coordinates were the old deck's, are dropped; ``references`` records
        the step and deck of every change, the deck a cold re-solve of the later steps starts from); otherwise
        the trial fails on the old deck.
        """
        old = {name: self.__dict__[name] for name in self._BOUND}
        try:
            model = self.model.rereferenced(anchor["b"])
        except (VmecError, RuntimeError):
            return None
        self._bind(model, x)
        fit = None
        try:
            fit = model.solve_boundary(self._params_at(jnp.asarray(x)), self.field(jnp.asarray(x)),
                                       max_nfev=self.boundary_max_nfev, continuation=False)
        except (VmecError, RuntimeError):
            pass
        if fit is None or not fit["converged"]:
            if fit is not None:
                self.stats["boundary_evaluations"] += fit["nfev"]
            self.__dict__.update(old)
            return None
        self.stats["rereferences"] = self.stats.get("rereferences", 0) + 1
        self.references.append(dict(step=self.accepted_step, trial=self.stats["trials"], deck=model.fixed))
        self._anchor = {}
        self._cache = {k: v for k, v in self._cache.items() if v is not None and "gradients" in v}
        return fit

    def _store(self, x, fit):
        """Cache the converged ``fit`` as the trial at ``x``."""
        self.stats["boundary_evaluations"] += fit["nfev"]
        state, mask, params_b, _ = fit["aux"]
        self._cache[x.tobytes()] = entry = dict(fit, state=state, mask=mask, params=params_b,
                                                values=np.asarray(self._rows(state, params_b, jnp.asarray(x))))
        return entry

    def _record(self, x):
        sol = self._solve(self._x(x))
        if sol is None:
            self.metadata["holder"]["failed_trials"] += 1
            raise TrialRejected("no three-terms free boundary at this design point")
        return sol

    def _linearize(self, x, sol):
        """Every row's design gradient at ``sol`` (cached), re-solved tight and polished with its own Jacobian."""
        if "gradients" in sol:
            return sol["gradients"]
        model, xj = self.model, jnp.asarray(x)
        field, nb = self.field(xj), model.x0.size
        if self._warm_threads[0] is not None:  # the response fallbacks: wait instead of compiling them twice
            self._warm_threads[0].join()

        def relinearize():
            extra = self._plasma_directions(xj) if self._plasma.size else None
            try:
                J, _, _, aux = model.linearize(sol["x"], sol["aux"], field, extra=extra,
                                               extra_scale=None if extra is None else self.scales[self._plasma])
            except RuntimeError as error:  # the trial's state does not tighten to the deck's tolerance
                raise TrialRejected(str(error)) from error
            state, mask, params, _ = aux
            sol.update(aux=aux, state=state, mask=mask, params=params,
                       rows=np.asarray(model._rows(state, params, model.attach(field))),
                       values=np.asarray(self._rows(state, params, xj)))
            self.stats["linearizations"] += 1
            return J

        J = relinearize()
        # to this Jacobian's fixed point, solved tight; J stays the anchor of the next trials
        params_x = self._params_at(xj)
        polished = model.solve_boundary(params_x, field, x0=sol["x"], jacobian=J[:, :nb],
                                        max_nfev=self.boundary_max_nfev)
        self.stats["boundary_evaluations"] += polished["nfev"]
        if not polished["converged"]:
            raise TrialRejected("the polish of the boundary did not converge")
        if polished["njev"]:  # J's own steps stalled and fresh ones finished: the anchor is relinearized there
            sol.update(x=polished["x"], aux=polished["aux"])
            J = relinearize()
        else:
            tight = model.evaluate(polished["x"], params_x, field, tight=True)
            if tight is None:
                raise TrialRejected("the polished boundary has no certified equilibrium")
            rows, aux = tight
            state, mask, params, _ = aux
            sol.update(x=polished["x"], aux=aux, state=state, mask=mask, params=params, rows=rows,
                       values=np.asarray(self._rows(state, params, xj)))
        J_b, J_p = J[:, :nb], J[:, nb:]
        if self._warm_threads[1] is not None:  # the pullback and direct Jacobian
            self._warm_threads[1].join()
        state, mask, params = model._im._canonical_arrays(model.cfg, (sol["state"], sol["mask"], sol["params"]))
        factors = model._im.block_factors_at(model.cfg, params, state, mask)
        params_bar = self._pullback(state, mask, params, xj, factors)
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        Q, R = np.linalg.qr(J_b)
        lam = Q @ np.linalg.solve(R.T, G_b.T)  # residual-space multipliers, a column per row
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
        gradients -= np.stack([np.asarray(self._field_vjp(state, params, xj, jnp.asarray(column), model.deck)) for column in lam.T])
        gradients[:, self._plasma] -= lam.T @ J_p
        self._anchor = dict(x=self._x(x).copy(), b=sol["x"], J_b=J_b, J_p=J_p, state=state,
                            params=params, cost=0.5 * sol["rows"] @ sol["rows"])
        sol["gradients"] = gradients
        return gradients

    def _pullback(self, state, mask, params, x, factors):
        """Every row's ``ImplicitParams`` gradient, each through its own function's reverse pass only.

        ``factors``: the block factors at ``state`` (:func:`~vmex.core.implicit.block_factors_at`, an executable
        compiled once per config), not a second block assembly inside these programs.
        """
        pieces = [program(state, params, x) for program in self._cotangents]
        state_bars, params_bars = jax.tree.map(lambda *a: jnp.concatenate(a), *pieces)
        return self._adjoint(state, mask, params, state_bars, params_bars, factors)

    def _warm_gradient_programs(self, x0):
        """Compile the design gradient's programs on background threads while the seed design is solved.

        Two daemon threads, from placeholder arguments of the model's structure (values do not enter a
        compilation): the state-response fallbacks the plasma-parameter columns need (GMRES corrector and
        refinement, :func:`~vmex.core.implicit.warm_staged_tangent`), joined before the first relinearization,
        and the pullback through the equilibrium (each function's cotangent program and the shared adjoint, compiled
        in parallel) and the direct design Jacobian, joined before the first pullback.  Cache warming only: the
        same executables; a failure leaves the real call to compile.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        model = self.model
        im, cfg, params0 = model._im, model.cfg, model.params0
        if not im._overlapping_compiles_pay():
            return None, None

        def tangents():
            try:
                im.warm_staged_tangent(cfg, params0, chunk=model.chunk, certify_rtol=cfg.adjoint_tol,
                                       certify_maxiter=cfg.adjoint_maxiter, fallbacks=True)
            except Exception:  # cache warming only: the real calls compile on demand
                pass

        def gradient():
            try:
                state, mask, params = im._canonical_arrays(cfg, im._placeholder_arguments(cfg, params0))
                arguments = (state, params, state, mask)
                factors = im._zeros_like_structure(
                    cfg, jax.eval_shape(lambda *a: im._refine_block_factor_core(*a, cfg), *arguments))
                x = jnp.asarray(np.zeros_like(np.asarray(x0, dtype=float)))
                pieces = [jax.eval_shape(program, state, params, x) for program in self._cotangents]
                bars = jax.tree.map(lambda *a: jax.ShapeDtypeStruct((sum(b.shape[0] for b in a),) + a[0].shape[1:],
                                                                    a[0].dtype), *pieces)
                programs = [lambda program=program: program.lower(state, params, x).compile()
                            for program in self._cotangents]
                programs += [lambda: self._adjoint.lower(state, mask, params, *bars, factors).compile(),
                             lambda: self._direct.lower(state, params, x).compile()]
                with ThreadPoolExecutor(max_workers=len(programs)) as pool:  # XLA compiles release the GIL
                    for done in [pool.submit(compile_) for compile_ in programs]:
                        done.result()
            except Exception:  # cache warming only
                pass

        threads = tuple(threading.Thread(target=work, name=name, daemon=True)
                        for work, name in ((tangents, "vmex-tangent-fallback-warm"), (gradient, "vmex-gradient-warm")))
        for thread in threads:
            thread.start()
        return threads

    def _value(self, x):
        return float(self._record(x)["values"][0])

    def _value_gradient(self, x):
        sol = self._record(x)
        gradients = self._linearize(x, sol)
        return float(sol["values"][0]), gradients[0].copy()

    def constraint_values(self, x):
        """The quantities' values, in their units."""
        return self._record(x)["values"][1:].copy()

    def constraint_jac(self, x):
        """The quantities' design gradients."""
        return self._linearize(x, self._record(x))[1:].copy()

    def nonlinear_constraint(self, lower, upper, *, scales=1.0):
        """Bound the quantities supplied to ``from_loss``, in their units."""
        from .freeboundary_problem import _nonlinear_constraint

        return _nonlinear_constraint(self.constraint_values, self.constraint_jac, lower, upper, scales)

    def accept_x(self, x):
        """Promote an evaluated point after the optimizer accepts it; drop the other trials."""
        x = self._x(x)
        key = x.tobytes()
        if self._cache.get(key) is None:
            raise ValueError("evaluate the candidate before accepting it")
        self.accepted, self.accepted_step = _AcceptedPoint(x.copy()), self.accepted_step + 1
        self._cache = {key: self._cache[key]}

    def state(self, x):
        """``(state, params)`` of ``x``'s free boundary (``params`` the ``ImplicitParams`` it was solved at)."""
        sol = self._record(x)
        return sol["state"], sol["params"]

    def boundary_residual(self, x) -> BoundaryResidual:
        """The three interface conditions at ``x``'s free boundary."""
        sol = self._record(x)
        return self.model.boundary_residual(sol["state"], sol["params"], self.field(jnp.asarray(x)))

    def boundary_coordinates(self, x):
        """The free boundary of design ``x`` in the deck model's coordinates (and current values): what
        ``from_loss(boundary_start=)`` restarts from on the same deck."""
        return np.array(self._record(x)["x"], dtype=float)

    def equilibrium_from_x(self, x):
        """The free-boundary equilibrium of ``x`` as an ``Equilibrium`` (fixed-boundary solve on its boundary)."""
        from .implicit import input_with_params
        from .optimize import solve_equilibrium

        sol = self._record(x)
        return solve_equilibrium(input_with_params(self.model.fixed, sol["params"]), restart_from=sol["state"])

    def close(self):
        """Drop every cached trial except the accepted one."""
        key = self.accepted.parameters.tobytes()
        self._cache = {k: v for k, v in self._cache.items() if k == key}
