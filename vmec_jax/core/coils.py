"""Fourier coil sets and pure-JAX Biot-Savart external fields (plan.md §8).

This module is the clean-core home of direct-coil free-boundary inputs:

- :class:`CoilSet` — a frozen pytree of Fourier-represented coil centerlines
  plus currents (differentiable leaves: ``base_curve_dofs``,
  ``base_currents``), ported from the legacy parity-proven
  ``vmec_jax.external_fields.coils_jax.CoilFieldParams``.
- :func:`biot_savart` / :func:`b_cyl` — pure-JAX Biot-Savart evaluation at
  Cartesian / cylindrical points, jit- and grad-compatible with respect to
  coil dofs and currents.
- :func:`field_on_cylindrical_grid` / :func:`to_mgrid_data` — sample the coil
  field on a VMEC cylindrical grid and package it as
  :class:`vmec_jax.core.mgrid.MgridData` so :func:`vmec_jax.core.mgrid
  .write_mgrid` produces a VMEC2000-compatible mgrid file (the equivalent of
  ESSOS ``coils_to_mgrid``, uwplasma/ESSOS PR#33).

Conventions (intentionally identical to ESSOS)
----------------------------------------------
Fourier coefficients, per Cartesian component, for curve parameter
``t in [0, 1)``:

``dofs[..., 0]``
    constant term.
``dofs[..., 2*k-1]``
    coefficient multiplying ``sin(2*pi*k*t)``.
``dofs[..., 2*k]``
    coefficient multiplying ``cos(2*pi*k*t)``.

Biot-Savart normalization (ESSOS ``essos.fields.BiotSavart.B``):

``B(x) = 1e-7 * sum_coils I_coil * mean_t( gamma'(t) x (x - gamma(t))
/ |x - gamma(t)|^3 )``

i.e. the curve quadrature is the *mean* over ``n_segments`` uniform points of
the normalized parameter ``t`` (``gamma' = d gamma / d t`` carries the
``2*pi`` factors), and ``1e-7 = mu0 / (4*pi)``.  Stellarator-symmetry images
carry the opposite current; symmetry expansion ordering is
``for k in range(nfp): for flip in (False, True[stellsym]): all base coils``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi as _PI
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .mgrid import MgridData

_TWO_PI = 2.0 * _PI

#: mu0 / (4 pi) — the ESSOS-established Biot-Savart prefactor.
_BIOT_SAVART_PREFACTOR = 1.0e-7


# ---------------------------------------------------------------------------
# CoilSet pytree
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoilSet:
    """Fourier-represented coils + currents (frozen JAX pytree).

    A pytree: ``base_curve_dofs`` and ``base_currents`` are differentiable
    leaves; everything else is static metadata, so instances flow through
    ``jax.jit`` / ``jax.grad`` unchanged.

    Parameters
    ----------
    base_curve_dofs:
        Array with shape ``(n_base_coils, 3, 2 * order + 1)``.  Second axis
        is Cartesian ``x, y, z``; last axis follows the ESSOS Fourier
        convention documented in this module.
    base_currents:
        Array with shape ``(n_base_coils,)``, multiplied by
        ``current_scale`` before Biot-Savart evaluation (ESSOS stores
        normalized ``dofs_currents`` plus a ``currents_scale``).
    n_segments:
        Number of uniform curve quadrature points per coil.
    nfp:
        Number of field periods for rotational symmetry expansion.
    stellsym:
        Whether to add stellarator-symmetry reflected coils (opposite
        current, matching ESSOS).
    current_scale:
        Static scalar multiplying ``base_currents``.
    regularization_epsilon:
        Optional distance smoothing in the Biot-Savart denominator.
    chunk_size:
        Optional point chunk size for memory-limited field evaluation
        (``jax.lax.map(..., batch_size=chunk_size)``).
    """

    base_curve_dofs: Any
    base_currents: Any
    n_segments: int
    nfp: int = 1
    stellsym: bool = False
    current_scale: float = 1.0
    regularization_epsilon: float = 0.0
    chunk_size: int | None = None

    # -- static shape helpers ------------------------------------------------

    @property
    def n_base_coils(self) -> int:
        """Number of independent (pre-symmetry) coils."""

        return int(jnp.shape(self.base_curve_dofs)[0])

    @property
    def order(self) -> int:
        """Fourier order of the curve representation."""

        return (int(jnp.shape(self.base_curve_dofs)[2]) - 1) // 2

    @property
    def n_coils(self) -> int:
        """Total coil count after nfp/stellsym symmetry expansion."""

        return self.n_base_coils * int(self.nfp) * (2 if self.stellsym else 1)

    def with_arrays(
        self,
        *,
        base_curve_dofs: Any | None = None,
        base_currents: Any | None = None,
    ) -> "CoilSet":
        """Return a copy with updated differentiable leaves."""

        return replace(
            self,
            base_curve_dofs=self.base_curve_dofs if base_curve_dofs is None else base_curve_dofs,
            base_currents=self.base_currents if base_currents is None else base_currents,
        )

    # -- constructors ----------------------------------------------------------

    @classmethod
    def from_essos(
        cls,
        coils: Any,
        regularization_epsilon: float = 0.0,
        chunk_size: int | None = None,
    ) -> "CoilSet":
        """Convert an ESSOS ``Coils`` object into a :class:`CoilSet`.

        ESSOS is intentionally not imported here; the adapter duck-types the
        ESSOS ``Coils`` attributes ``dofs_curves``, ``dofs_currents``,
        ``currents_scale``, ``n_segments``, ``nfp``, and ``stellsym``.

        Raises
        ------
        ImportError
            If the supplied object does not expose the expected ESSOS
            attributes (e.g. ESSOS is not installed and something else was
            passed).
        """

        required = ("dofs_curves", "dofs_currents", "currents_scale", "n_segments", "nfp", "stellsym")
        missing = [name for name in required if not hasattr(coils, name)]
        if missing:
            raise ImportError(
                "CoilSet.from_essos expects an essos.coils.Coils instance "
                f"(got {type(coils).__name__} missing {', '.join(missing)}). "
                "Install ESSOS (pip install essos, or pip install -e <ESSOS checkout>) "
                "and pass essos.coils.Coils(curves, currents)."
            )

        base_curve_dofs = jnp.asarray(coils.dofs_curves)
        base_currents = jnp.asarray(coils.dofs_currents)
        if base_curve_dofs.ndim != 3 or base_curve_dofs.shape[1] != 3 or base_curve_dofs.shape[2] % 2 != 1:
            raise ValueError("ESSOS dofs_curves must have shape (n_base_coils, 3, 2 * order + 1)")
        if base_currents.ndim != 1:
            raise ValueError("ESSOS dofs_currents must have shape (n_base_coils,)")
        if base_currents.shape[0] != base_curve_dofs.shape[0]:
            raise ValueError(
                "ESSOS dofs_currents length must match dofs_curves n_base_coils: "
                f"{base_currents.shape[0]} != {base_curve_dofs.shape[0]}"
            )
        normalized_chunk = None if chunk_size is None else int(chunk_size)
        if normalized_chunk is not None and normalized_chunk <= 0:
            raise ValueError(f"chunk_size must be positive, got {normalized_chunk}")

        return cls(
            base_curve_dofs=base_curve_dofs,
            base_currents=base_currents,
            n_segments=int(coils.n_segments),
            nfp=int(coils.nfp),
            stellsym=bool(coils.stellsym),
            current_scale=float(coils.currents_scale),
            regularization_epsilon=float(regularization_epsilon),
            chunk_size=normalized_chunk,
        )

    # -- field evaluation (thin method aliases of the module functions) --------

    def b_xyz(self, points_xyz: Any) -> Any:
        """Alias for :func:`biot_savart`."""

        return biot_savart(self, points_xyz)

    def b_cyl(self, r: Any, phi: Any, z: Any) -> tuple[Any, Any, Any]:
        """Alias for :func:`b_cyl`."""

        return b_cyl(self, r, phi, z)


# Pytree registration: dofs/currents are leaves, everything else is static.
jax.tree_util.register_dataclass(
    CoilSet,
    data_fields=["base_curve_dofs", "base_currents"],
    meta_fields=["n_segments", "nfp", "stellsym", "current_scale", "regularization_epsilon", "chunk_size"],
)


# ---------------------------------------------------------------------------
# Fourier curve evaluation (ported verbatim from legacy coils_jax)
# ---------------------------------------------------------------------------


def _fourier_basis(n_segments: int, order: int) -> tuple[Any, Any, Any]:
    t = jnp.linspace(0.0, 1.0, int(n_segments), endpoint=False)
    k = jnp.arange(1, int(order) + 1, dtype=t.dtype)
    phase = _TWO_PI * t[:, None] * k[None, :]
    return t, jnp.sin(phase), jnp.cos(phase)


def _check_dofs(dofs: Any) -> int:
    if dofs.ndim != 3 or dofs.shape[1] != 3 or dofs.shape[2] % 2 != 1:
        raise ValueError("base_curve_dofs must have shape (n_base_coils, 3, 2 * order + 1)")
    return (int(dofs.shape[2]) - 1) // 2


def curves_gamma(base_curve_dofs: Any, n_segments: int) -> Any:
    """Evaluate Fourier curve centerlines: ``(n_base_coils, n_segments, 3)``."""

    dofs = jnp.asarray(base_curve_dofs)
    order = _check_dofs(dofs)
    _, sin_basis, cos_basis = _fourier_basis(n_segments, order)
    gamma = dofs[:, :, 0][:, None, :]
    if order == 0:
        return jnp.broadcast_to(gamma, (dofs.shape[0], int(n_segments), 3))
    sin_coeff = dofs[:, :, 1::2]
    cos_coeff = dofs[:, :, 2::2]
    gamma = gamma + jnp.einsum("nck,sk->nsc", sin_coeff, sin_basis)
    gamma = gamma + jnp.einsum("nck,sk->nsc", cos_coeff, cos_basis)
    return gamma


def curves_gamma_dash(base_curve_dofs: Any, n_segments: int) -> Any:
    """Evaluate ``d gamma / d t`` for normalized curve parameter ``t``."""

    dofs = jnp.asarray(base_curve_dofs)
    order = _check_dofs(dofs)
    if order == 0:
        return jnp.zeros((dofs.shape[0], int(n_segments), 3), dtype=dofs.dtype)
    t, sin_basis, cos_basis = _fourier_basis(n_segments, order)
    k = jnp.arange(1, order + 1, dtype=t.dtype)
    factor = _TWO_PI * k
    sin_coeff = dofs[:, :, 1::2]
    cos_coeff = dofs[:, :, 2::2]
    gamma_dash = jnp.einsum("nck,sk,k->nsc", sin_coeff, cos_basis, factor)
    gamma_dash = gamma_dash - jnp.einsum("nck,sk,k->nsc", cos_coeff, sin_basis, factor)
    return gamma_dash


def _rotation_reflection_matrix(phi: Any, flip: bool) -> Any:
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    rot = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T
    if flip:
        rot = rot @ jnp.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return rot


def _apply_symmetry_to_xyz_array(base: Any, *, nfp: int, stellsym: bool) -> Any:
    values = jnp.asarray(base)
    flip_list = (False, True) if bool(stellsym) else (False,)
    expanded = []
    for k in range(int(nfp)):
        angle = _TWO_PI * k / int(nfp)
        for flip in flip_list:
            mat = _rotation_reflection_matrix(angle, flip)
            expanded.append(jnp.einsum("...c,cd->...d", values, mat))
    return jnp.concatenate(expanded, axis=0)


def _apply_symmetry_to_currents(base_currents: Any, *, nfp: int, stellsym: bool) -> Any:
    currents = jnp.asarray(base_currents)
    expanded = []
    for _k in range(int(nfp)):
        expanded.append(currents)
        if bool(stellsym):
            expanded.append(-currents)
    return jnp.concatenate(expanded, axis=0)


def coil_geometry(coilset: CoilSet) -> tuple[Any, Any, Any]:
    """Return symmetry-expanded ``(gamma, gamma_dash, currents)``.

    ``gamma``/``gamma_dash`` have shape ``(n_coils, n_segments, 3)``;
    ``currents`` are physical (``current_scale`` applied, reflected images
    negated) with shape ``(n_coils,)``.
    """

    dofs = jnp.asarray(coilset.base_curve_dofs)
    base_gamma = curves_gamma(dofs, coilset.n_segments)
    base_gamma_dash = curves_gamma_dash(dofs, coilset.n_segments)
    gamma = _apply_symmetry_to_xyz_array(base_gamma, nfp=coilset.nfp, stellsym=coilset.stellsym)
    gamma_dash = _apply_symmetry_to_xyz_array(base_gamma_dash, nfp=coilset.nfp, stellsym=coilset.stellsym)
    currents = coilset.current_scale * _apply_symmetry_to_currents(
        coilset.base_currents, nfp=coilset.nfp, stellsym=coilset.stellsym
    )
    return gamma, gamma_dash, currents


# ---------------------------------------------------------------------------
# Biot-Savart (ported verbatim from legacy coils_jax.biot_savart_xyz)
# ---------------------------------------------------------------------------


def _biot_savart_xyz_vectorized(
    points_xyz: Any,
    gamma: Any,
    gamma_dash: Any,
    currents: Any,
    regularization_epsilon: float = 0.0,
) -> Any:
    points = jnp.asarray(points_xyz)
    original_shape = points.shape[:-1]
    flat = jnp.reshape(points, (-1, 3))
    rx = flat[None, None, :, 0] - gamma[:, :, None, 0]
    ry = flat[None, None, :, 1] - gamma[:, :, None, 1]
    rz = flat[None, None, :, 2] - gamma[:, :, None, 2]
    eps = jnp.asarray(regularization_epsilon, dtype=points.dtype)
    denom2 = rx * rx + ry * ry + rz * rz + eps * eps
    inv_r = jax.lax.rsqrt(denom2)
    inv_r3 = inv_r * inv_r * inv_r

    gx = gamma_dash[:, :, None, 0]
    gy = gamma_dash[:, :, None, 1]
    gz = gamma_dash[:, :, None, 2]
    weights = jnp.asarray(currents, dtype=points.dtype)[:, None, None] * inv_r3
    field_x = jnp.mean(jnp.sum(weights * (gy * rz - gz * ry), axis=0), axis=0)
    field_y = jnp.mean(jnp.sum(weights * (gz * rx - gx * rz), axis=0), axis=0)
    field_z = jnp.mean(jnp.sum(weights * (gx * ry - gy * rx), axis=0), axis=0)
    field = _BIOT_SAVART_PREFACTOR * jnp.stack((field_x, field_y, field_z), axis=-1)
    return jnp.reshape(field, original_shape + (3,))


def biot_savart_from_geometry(
    points_xyz: Any,
    gamma: Any,
    gamma_dash: Any,
    currents: Any,
    *,
    regularization_epsilon: float = 0.0,
    chunk_size: int | None = None,
) -> Any:
    """Evaluate the Biot-Savart field at Cartesian points from raw geometry.

    ``chunk_size`` limits peak memory by mapping over evaluation points in
    batches; the unchunked path is faster for small grids and is the default.
    """

    points = jnp.asarray(points_xyz)
    original_shape = points.shape[:-1]
    flat = jnp.reshape(points, (-1, 3))
    if chunk_size is None:
        return _biot_savart_xyz_vectorized(
            points,
            gamma,
            gamma_dash,
            currents,
            regularization_epsilon=regularization_epsilon,
        )

    def one_point(point: Any) -> Any:
        value = _biot_savart_xyz_vectorized(
            point[None, :],
            gamma,
            gamma_dash,
            currents,
            regularization_epsilon=regularization_epsilon,
        )
        return value[0]

    values = jax.lax.map(one_point, flat, batch_size=int(chunk_size))
    return jnp.reshape(values, original_shape + (3,))


def biot_savart(coilset: CoilSet, points_xyz: Any) -> Any:
    """Cartesian coil field ``B(x, y, z)`` with shape ``points.shape[:-1] + (3,)``.

    Pure JAX; differentiable with respect to ``coilset.base_curve_dofs`` and
    ``coilset.base_currents`` (and the evaluation points).
    """

    gamma, gamma_dash, currents = coil_geometry(coilset)
    return biot_savart_from_geometry(
        points_xyz,
        gamma,
        gamma_dash,
        currents,
        regularization_epsilon=coilset.regularization_epsilon,
        chunk_size=coilset.chunk_size,
    )


def _cylindrical_points_to_xyz(r: Any, phi: Any, z: Any) -> Any:
    rb, phib, zb = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
    return jnp.stack((rb * jnp.cos(phib), rb * jnp.sin(phib), zb), axis=-1)


def _xyz_field_to_cylindrical(b_xyz: Any, phi: Any) -> tuple[Any, Any, Any]:
    phib = jnp.broadcast_to(jnp.asarray(phi), b_xyz.shape[:-1])
    bx = b_xyz[..., 0]
    by = b_xyz[..., 1]
    bz = b_xyz[..., 2]
    br = bx * jnp.cos(phib) + by * jnp.sin(phib)
    bphi = -bx * jnp.sin(phib) + by * jnp.cos(phib)
    return br, bphi, bz


def b_cyl(coilset: CoilSet, r: Any, phi: Any, z: Any) -> tuple[Any, Any, Any]:
    """Cylindrical coil field ``(B_r, B_phi, B_z)`` at broadcastable points."""

    points = _cylindrical_points_to_xyz(r, phi, z)
    field_xyz = biot_savart(coilset, points)
    return _xyz_field_to_cylindrical(field_xyz, phi)


# ---------------------------------------------------------------------------
# mgrid sampling (equivalent of ESSOS coils_to_mgrid, PR#33)
# ---------------------------------------------------------------------------


def _grid_phi_z_r(
    rmin: float, rmax: float, zmin: float, zmax: float, ir: int, jz: int, kp: int, nfp: int
) -> tuple[Any, Any, Any]:
    """Cylindrical tensor grid in mgrid ``(phi, zee, rad)`` layout.

    R and Z are inclusive at both ends; phi spans ``[0, 2*pi/nfp)`` with the
    endpoint excluded — identical to MAKEGRID / ESSOS ``coils_to_mgrid``.
    """

    rs = jnp.linspace(float(rmin), float(rmax), int(ir), endpoint=True)
    zs = jnp.linspace(float(zmin), float(zmax), int(jz), endpoint=True)
    phis = jnp.linspace(0.0, _TWO_PI / int(nfp), int(kp), endpoint=False)
    return jnp.meshgrid(phis, zs, rs, indexing="ij")  # each (kp, jz, ir)


def field_on_cylindrical_grid(
    coilset: CoilSet,
    rmin: float,
    rmax: float,
    zmin: float,
    zmax: float,
    ir: int,
    jz: int,
    kp: int,
    nfp: int | None = None,
    *,
    per_unit_current: bool = False,
    single_group: bool = False,
) -> tuple[Any, Any, Any]:
    """Sample the coil field on a VMEC cylindrical grid, per coil-current group.

    Returns ``(br, bp, bz)``, each with shape ``(nextcur, kp, jz, ir)`` — the
    stacked mgrid layout.  A "coil-current group" is one base coil together
    with all of its nfp/stellsym symmetry images (which share its current up
    to sign), so ``nextcur == coilset.n_base_coils`` unless ``single_group``
    lumps everything into one group (``nextcur == 1``, the ESSOS
    ``coils_to_mgrid`` convention).

    ``per_unit_current`` divides each group's field by its physical current
    (mgrid_mode "S" scaling); otherwise raw fields with the physical currents
    baked in are returned (modes "R"/"N").
    """

    if single_group and per_unit_current:
        raise ValueError("per_unit_current is undefined for a single lumped coil group")
    nfp_eff = int(coilset.nfp if nfp is None else nfp)
    phi_g, z_g, r_g = _grid_phi_z_r(rmin, rmax, zmin, zmax, ir, jz, kp, nfp_eff)
    points = _cylindrical_points_to_xyz(r_g, phi_g, z_g)  # (kp, jz, ir, 3)

    gamma, gamma_dash, currents = coil_geometry(coilset)

    def field_cyl(cur: Any) -> tuple[Any, Any, Any]:
        b_xyz = biot_savart_from_geometry(
            points,
            gamma,
            gamma_dash,
            cur,
            regularization_epsilon=coilset.regularization_epsilon,
            chunk_size=coilset.chunk_size,
        )
        return _xyz_field_to_cylindrical(b_xyz, phi_g)

    if single_group:
        br, bp, bz = field_cyl(currents)
        return br[None], bp[None], bz[None]

    n_base = coilset.n_base_coils
    coil_ids = jnp.arange(gamma.shape[0]) % n_base  # symmetry blocks repeat base order
    phys_currents = coilset.current_scale * jnp.asarray(coilset.base_currents)
    br_groups, bp_groups, bz_groups = [], [], []
    for g in range(n_base):
        cur_g = jnp.where(coil_ids == g, currents, 0.0)
        if per_unit_current:
            cur_g = cur_g / phys_currents[g]
        br, bp, bz = field_cyl(cur_g)
        br_groups.append(br)
        bp_groups.append(bp)
        bz_groups.append(bz)
    return jnp.stack(br_groups), jnp.stack(bp_groups), jnp.stack(bz_groups)


def to_mgrid_data(
    coilset: CoilSet,
    rmin: float,
    rmax: float,
    zmin: float,
    zmax: float,
    ir: int,
    jz: int,
    kp: int,
    nfp: int | None = None,
    *,
    mgrid_mode: str = "S",
    single_group: bool = False,
    coil_group_names: tuple[str, ...] | None = None,
) -> MgridData:
    """Package the coil field as :class:`vmec_jax.core.mgrid.MgridData`.

    ``vmec_jax.core.mgrid.write_mgrid(path, to_mgrid_data(...))`` produces a
    VMEC2000-compatible mgrid netCDF file — the equivalent of ESSOS
    ``coils_to_mgrid`` (uwplasma/ESSOS PR#33).

    mgrid_mode conventions:

    - ``"S"`` (default, MAKEGRID "scaled"): one group per base coil, fields
      per unit current, ``raw_coil_cur`` set to the physical currents; VMEC
      reproduces this coil set with ``EXTCUR == raw_coil_cur``.
    - ``"R"`` (raw): per-group fields with physical currents baked in,
      ``raw_coil_cur`` set to the physical currents.
    - ``"N"`` (none/raw, the ESSOS writer convention): raw fields and
      ``raw_coil_cur`` of ones; combine with ``single_group=True`` to
      reproduce ESSOS ``coils_to_mgrid`` output exactly (one lumped group).
    """

    mode = str(mgrid_mode).strip().upper()[:1] or "S"
    if mode not in ("S", "R", "N"):
        raise ValueError(f"mgrid_mode must be one of 'S', 'R', 'N'; got {mgrid_mode!r}")
    if single_group and mode == "S":
        raise ValueError("mgrid_mode 'S' (per unit current) requires per-coil groups; use 'N' or 'R'")

    nfp_eff = int(coilset.nfp if nfp is None else nfp)
    br, bp, bz = field_on_cylindrical_grid(
        coilset,
        rmin,
        rmax,
        zmin,
        zmax,
        ir,
        jz,
        kp,
        nfp_eff,
        per_unit_current=(mode == "S"),
        single_group=single_group,
    )

    nextcur = 1 if single_group else coilset.n_base_coils
    phys_currents = np.asarray(coilset.current_scale * jnp.asarray(coilset.base_currents), dtype=np.float64)
    if single_group:
        raw_coil_cur: tuple[float, ...] = (1.0,)
    elif mode == "N":
        raw_coil_cur = tuple(1.0 for _ in range(nextcur))
    else:
        raw_coil_cur = tuple(float(c) for c in phys_currents)

    if coil_group_names is None:
        names = ("coil_set",) if single_group else tuple(f"coil_{g + 1:03d}" for g in range(nextcur))
    else:
        names = tuple(str(s) for s in coil_group_names)
        if len(names) != nextcur:
            raise ValueError(f"coil_group_names length {len(names)} != nextcur {nextcur}")

    return MgridData(
        rmin=float(rmin),
        rmax=float(rmax),
        zmin=float(zmin),
        zmax=float(zmax),
        ir=int(ir),
        jz=int(jz),
        kp=int(kp),
        nfp=nfp_eff,
        nextcur=nextcur,
        mgrid_mode=mode,
        coil_groups=names,
        raw_coil_cur=raw_coil_cur,
        br=np.asarray(br, dtype=np.float64),
        bp=np.asarray(bp, dtype=np.float64),
        bz=np.asarray(bz, dtype=np.float64),
    )


__all__ = [
    "CoilSet",
    "biot_savart",
    "biot_savart_from_geometry",
    "b_cyl",
    "coil_geometry",
    "curves_gamma",
    "curves_gamma_dash",
    "field_on_cylindrical_grid",
    "to_mgrid_data",
]
