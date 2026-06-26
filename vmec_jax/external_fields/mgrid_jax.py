"""Differentiable JAX interpolation for VMEC mgrid fields.

This backend is a compatibility layer for VMEC2000-style mgrid data.  It is
not intended to replace direct coil fields for single-stage optimization, but
it provides differentiable tests with respect to field values, external-current
weights, and evaluation coordinates away from grid-cell boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from vmec_jax._compat import jnp, tree_util


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MGridFieldParams:
    """JAX mgrid field values and interpolation metadata.

    Field arrays use VMEC/mgrid layout ``(nextcur, kp, jz, ir)``.
    """

    br: Any
    bphi: Any
    bz: Any
    extcur: Any
    rmin: float
    rmax: float
    zmin: float
    zmax: float
    nfp: int = 1
    use_vmec_kv: bool = False

    def tree_flatten(self):
        """Return JAX pytree leaves and static metadata for transformations."""
        children = (self.br, self.bphi, self.bz, self.extcur)
        aux = (
            float(self.rmin),
            float(self.rmax),
            float(self.zmin),
            float(self.zmax),
            int(self.nfp),
            bool(self.use_vmec_kv),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild the object from JAX pytree metadata and leaves."""
        rmin, rmax, zmin, zmax, nfp, use_vmec_kv = aux
        br, bphi, bz, extcur = children
        return cls(
            br=br,
            bphi=bphi,
            bz=bz,
            extcur=extcur,
            rmin=rmin,
            rmax=rmax,
            zmin=zmin,
            zmax=zmax,
            nfp=nfp,
            use_vmec_kv=use_vmec_kv,
        )

    def with_arrays(
        self,
        *,
        br: Any | None = None,
        bphi: Any | None = None,
        bz: Any | None = None,
        extcur: Any | None = None,
    ) -> "MGridFieldParams":
        """Return a copy with updated differentiable leaves."""

        return replace(
            self,
            br=self.br if br is None else br,
            bphi=self.bphi if bphi is None else bphi,
            bz=self.bz if bz is None else bz,
            extcur=self.extcur if extcur is None else extcur,
        )


def _check_field_shapes(br: Any, bphi: Any, bz: Any, extcur: Any) -> tuple[int, int, int, int]:
    shape = tuple(jnp.shape(br))
    if len(shape) != 4:
        raise ValueError("mgrid fields must have shape (nextcur, kp, jz, ir)")
    if tuple(jnp.shape(bphi)) != shape or tuple(jnp.shape(bz)) != shape:
        raise ValueError("br, bphi, and bz mgrid fields must have identical shapes")
    if int(jnp.shape(extcur)[0]) != int(shape[0]):
        raise ValueError(f"extcur length {jnp.shape(extcur)[0]} does not match nextcur {shape[0]}")
    nextcur, kp, jz, ir = (int(v) for v in shape)
    if ir < 2 or jz < 2 or kp < 1:
        raise ValueError(f"mgrid dimensions too small for interpolation: ir={ir} jz={jz} kp={kp}")
    return nextcur, kp, jz, ir


def _corner_values(field: Any, k0: Any, k1: Any, j0: Any, j1: Any, i0: Any, i1: Any) -> tuple[Any, ...]:
    f = jnp.asarray(field)
    return (
        f[:, k0, j0, i0],
        f[:, k0, j0, i1],
        f[:, k0, j1, i0],
        f[:, k0, j1, i1],
        f[:, k1, j0, i0],
        f[:, k1, j0, i1],
        f[:, k1, j1, i0],
        f[:, k1, j1, i1],
    )


def interpolate_mgrid_bfield_jax(
    br: Any,
    bphi: Any,
    bz: Any,
    *,
    extcur: Any,
    r: Any,
    z: Any,
    phi: Any,
    rmin: float,
    rmax: float,
    zmin: float,
    zmax: float,
    nfp: int = 1,
    use_vmec_kv: bool = False,
) -> tuple[Any, Any, Any]:
    """Trilinearly interpolate mgrid cylindrical field components.

    The interpolation matches the legacy linear mgrid layout for synthetic
    fields and supports differentiation with respect to field values, `extcur`,
    and coordinates away from cell boundaries.
    """

    nextcur, kp, jz, ir = _check_field_shapes(br, bphi, bz, extcur)
    del nextcur
    rr, zz, pp = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(z), jnp.asarray(phi))
    out_shape = rr.shape
    r_flat = jnp.clip(jnp.reshape(rr, (-1,)), float(rmin), float(rmax))
    z_flat = jnp.clip(jnp.reshape(zz, (-1,)), float(zmin), float(zmax))

    fr = (r_flat - float(rmin)) * ((ir - 1) / (float(rmax) - float(rmin)))
    fz = (z_flat - float(zmin)) * ((jz - 1) / (float(zmax) - float(zmin)))
    i0 = jnp.clip(jnp.floor(fr).astype(jnp.int32), 0, ir - 2)
    j0 = jnp.clip(jnp.floor(fz).astype(jnp.int32), 0, jz - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    wr = fr - i0
    wz = fz - j0

    if bool(use_vmec_kv):
        if rr.ndim == 0:
            raise ValueError("use_vmec_kv=True requires array inputs with an explicit zeta axis")
        nzeta = int(rr.shape[-1]) if int(rr.shape[-1]) > 0 else kp
        if kp == 1:
            k_idx = jnp.zeros(nzeta, dtype=jnp.int32)
        else:
            if nzeta < 1:
                raise ValueError("use_vmec_kv=True requires at least one zeta plane")
            if kp % nzeta != 0:
                raise ValueError(
                    "use_vmec_kv=True requires the number of mgrid zeta planes "
                    "to be divisible by the VMEC zeta axis length; kp must be divisible by nzeta"
                )
            # VMEC becoil samples the mgrid planes corresponding to the VMEC
            # zeta grid without toroidal interpolation.
            k_idx = jnp.arange(nzeta, dtype=jnp.int32) * int(kp // nzeta)
        k0 = jnp.broadcast_to(k_idx.reshape((1,) * (rr.ndim - 1) + (nzeta,)), rr.shape).reshape(-1)
        k1 = k0
        wk = jnp.zeros_like(fr)
    else:
        period = (2.0 * jnp.pi) / max(1, int(nfp))
        phi_flat = jnp.mod(jnp.reshape(pp, (-1,)), period)
        fk = phi_flat * (kp / period)
        k_floor = jnp.floor(fk)
        k0 = k_floor.astype(jnp.int32) % kp
        k1 = (k0 + 1) % kp
        wk = fk - k_floor

    w0r = 1.0 - wr
    w0z = 1.0 - wz
    w0k = 1.0 - wk
    cur = jnp.reshape(jnp.asarray(extcur), (-1, 1))

    def interp_one(field):
        """Evaluate interp one for external magnetic-field sampling for coils and mgrid data."""
        v000, v001, v010, v011, v100, v101, v110, v111 = _corner_values(field, k0, k1, j0, j1, i0, i1)
        c00 = v000 * w0r + v001 * wr
        c01 = v010 * w0r + v011 * wr
        c10 = v100 * w0r + v101 * wr
        c11 = v110 * w0r + v111 * wr
        c0 = c00 * w0z + c01 * wz
        c1 = c10 * w0z + c11 * wz
        c = c0 * w0k + c1 * wk
        return jnp.reshape(jnp.sum(cur * c, axis=0), out_shape)

    return interp_one(br), interp_one(bphi), interp_one(bz)


def sample_mgrid_field_cylindrical(params: MGridFieldParams, R: Any, Z: Any, phi: Any) -> tuple[Any, Any, Any]:
    """Sample an ``MGridFieldParams`` instance at cylindrical coordinates."""

    return interpolate_mgrid_bfield_jax(
        params.br,
        params.bphi,
        params.bz,
        extcur=params.extcur,
        r=R,
        z=Z,
        phi=phi,
        rmin=params.rmin,
        rmax=params.rmax,
        zmin=params.zmin,
        zmax=params.zmax,
        nfp=params.nfp,
        use_vmec_kv=params.use_vmec_kv,
    )


def mgrid_field_params_from_coils(
    coil_params: Any,
    *,
    rmin: float,
    rmax: float,
    zmin: float,
    zmax: float,
    nr: int,
    nz: int,
    nphi: int,
    nfp: int | None = None,
    extcur: tuple[float, ...] = (1.0,),
    use_vmec_kv: bool = True,
    regularization_epsilon: float | None = None,
    chunk_size: int | None = None,
) -> tuple[MGridFieldParams, np.ndarray, np.ndarray, np.ndarray]:
    """Sample direct coils on a VMEC ``mgrid`` tensor grid.

    The returned field arrays have one coil group that already contains the
    fully scaled direct-coil field.  Use ``EXTCUR = [1.0]`` when writing a VMEC
    input deck against this grid.
    """

    from .coils_jax import build_coil_field_geometry, sample_coil_field_cylindrical_from_geometry

    nr = int(nr)
    nz = int(nz)
    nphi = int(nphi)
    if nr < 2 or nz < 2 or nphi < 1:
        raise ValueError("nr and nz must be >=2 and nphi must be >=1")
    rmin = float(rmin)
    rmax = float(rmax)
    zmin = float(zmin)
    zmax = float(zmax)
    if not (np.isfinite(rmin) and np.isfinite(rmax) and rmax > rmin):
        raise ValueError("rmin/rmax must be finite with rmax > rmin")
    if not (np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin):
        raise ValueError("zmin/zmax must be finite with zmax > zmin")

    nfp_eff = int(getattr(coil_params, "nfp", 1) if nfp is None else nfp)
    if nfp_eff <= 0:
        raise ValueError("nfp must be positive")
    r_grid = np.linspace(rmin, rmax, nr, dtype=float)
    z_grid = np.linspace(zmin, zmax, nz, dtype=float)
    phi_grid = np.arange(nphi, dtype=float) * ((2.0 * np.pi / nfp_eff) / nphi)
    phi_mesh, z_mesh, r_mesh = np.meshgrid(phi_grid, z_grid, r_grid, indexing="ij")

    geometry = build_coil_field_geometry(coil_params)
    eps = float(getattr(coil_params, "regularization_epsilon", 0.0) if regularization_epsilon is None else regularization_epsilon)
    chunks = getattr(coil_params, "chunk_size", None) if chunk_size is None else chunk_size
    br, bphi, bz = sample_coil_field_cylindrical_from_geometry(
        geometry,
        r_mesh,
        z_mesh,
        phi_mesh,
        regularization_epsilon=eps,
        chunk_size=None if chunks is None else int(chunks),
    )
    params = MGridFieldParams(
        br=np.asarray(br, dtype=float)[None, ...],
        bphi=np.asarray(bphi, dtype=float)[None, ...],
        bz=np.asarray(bz, dtype=float)[None, ...],
        extcur=np.asarray(extcur, dtype=float),
        rmin=rmin,
        rmax=rmax,
        zmin=zmin,
        zmax=zmax,
        nfp=nfp_eff,
        use_vmec_kv=bool(use_vmec_kv),
    )
    return params, r_grid, z_grid, phi_grid


def write_mgrid_from_coils(
    path: str | Path,
    coil_params: Any,
    *,
    rmin: float,
    rmax: float,
    zmin: float,
    zmax: float,
    nr: int,
    nz: int,
    nphi: int,
    nfp: int | None = None,
    regularization_epsilon: float | None = None,
    chunk_size: int | None = None,
    mgrid_mode: str = "S",
    coil_group: str = "direct",
) -> Path:
    """Write a VMEC2000-compatible ``mgrid.nc`` sampled from direct coils."""

    try:
        import netCDF4
    except Exception as exc:  # pragma: no cover - optional dependency guard.
        raise ImportError("netCDF4 is required to write VMEC mgrid files") from exc

    params, _r_grid, _z_grid, _phi_grid = mgrid_field_params_from_coils(
        coil_params,
        rmin=rmin,
        rmax=rmax,
        zmin=zmin,
        zmax=zmax,
        nr=nr,
        nz=nz,
        nphi=nphi,
        nfp=nfp,
        extcur=(1.0,),
        use_vmec_kv=True,
        regularization_epsilon=regularization_epsilon,
        chunk_size=chunk_size,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    label = str(coil_group)[:30] or "direct"
    mode = (str(mgrid_mode).strip().upper() or "S")[0]
    with netCDF4.Dataset(str(path), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", max(8, len(label)))
        ds.createDimension("external_coil_groups", 1)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 1)
        ds.createDimension("rad", int(nr))
        ds.createDimension("zee", int(nz))
        ds.createDimension("phi", int(nphi))
        for name, value in (
            ("ir", int(nr)),
            ("jz", int(nz)),
            ("kp", int(nphi)),
            ("nfp", int(params.nfp)),
            ("nextcur", 1),
        ):
            ds.createVariable(name, "i4", ()).assignValue(value)
        for name, value in (
            ("rmin", float(rmin)),
            ("rmax", float(rmax)),
            ("zmin", float(zmin)),
            ("zmax", float(zmax)),
        ):
            ds.createVariable(name, "f8", ()).assignValue(value)
        mode_var = ds.createVariable("mgrid_mode", "S1", ("dim_00001",))
        mode_var[:] = np.asarray(list(mode), dtype="S1")
        group_var = ds.createVariable("coil_group", "S1", ("external_coil_groups", "stringsize"))
        group_arr = np.full((1, len(ds.dimensions["stringsize"])), b" ", dtype="S1")
        encoded = np.asarray(list(label.ljust(group_arr.shape[1])), dtype="S1")
        group_arr[0, :] = encoded
        group_var[:] = group_arr
        ds.createVariable("raw_coil_cur", "f8", ("external_coils",))[:] = np.asarray([1.0], dtype=float)
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = np.asarray(params.br[0], dtype=float)
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = np.asarray(params.bphi[0], dtype=float)
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = np.asarray(params.bz[0], dtype=float)
    return path
