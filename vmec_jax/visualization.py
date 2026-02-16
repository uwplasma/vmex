"""Visualization helpers (VTK export, field-line traces).

This module is intentionally dependency-free: it writes a small subset of VTK XML
formats in ASCII for easy inspection in ParaView:

- ``.vts``: VTK XML StructuredGrid (good for (theta,zeta) surfaces)
- ``.vtp``: VTK XML PolyData (good for field-line polylines)

The functions here are **not** performance critical; they are meant for examples
and debugging. They operate on NumPy arrays (not JAX) on purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _vtk_float_array(name: str | None, data: np.ndarray, *, n_comp: int) -> str:
    data = np.asarray(data)
    if n_comp == 1:
        flat = data.reshape(-1)
    else:
        flat = data.reshape(-1, n_comp)
    # ASCII output for simplicity.
    if n_comp == 1:
        body = " ".join(f"{x:.16e}" for x in flat)
    else:
        body = "\n".join(" ".join(f"{x:.16e}" for x in row) for row in flat)
    name_attr = f' Name="{name}"' if name is not None else ""
    return (
        f'<DataArray type="Float64"{name_attr} NumberOfComponents="{n_comp}" format="ascii">\n'
        f"{body}\n"
        "</DataArray>\n"
    )


def _vtk_int_array(name: str, data: np.ndarray) -> str:
    data = np.asarray(data, dtype=np.int32).reshape(-1)
    body = " ".join(str(int(x)) for x in data)
    return (
        f'<DataArray type="Int32" Name="{name}" NumberOfComponents="1" format="ascii">\n'
        f"{body}\n"
        "</DataArray>\n"
    )


def write_vts_structured_grid(
    path: str | Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    point_data: Dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a VTK XML StructuredGrid (``.vts``) from point coordinates.

    Parameters
    ----------
    x, y, z:
        Arrays of identical shape ``(nx, ny)`` or ``(nx, ny, nz)``.
        For a single surface, use ``(ntheta, nzeta)``.
    point_data:
        Optional dict of named arrays stored as PointData. Each array must have
        either shape matching ``x`` (scalar) or shape ``x.shape + (3,)`` (vector).
    """
    path = _as_path(path)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(f"x,y,z must have identical shapes, got {x.shape}, {y.shape}, {z.shape}")

    if x.ndim == 2:
        nx, ny = x.shape
        nz = 1
    elif x.ndim == 3:
        nx, ny, nz = x.shape
    else:
        raise ValueError(f"x must be 2D or 3D, got shape {x.shape}")

    pts = np.stack([x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")], axis=-1)

    point_data = point_data or {}

    # VTK extents are inclusive indices.
    extent = f"0 {nx-1} 0 {ny-1} 0 {nz-1}"

    pd_xml = ""
    if point_data:
        pieces = []
        for name, arr in point_data.items():
            a = np.asarray(arr)
            if a.shape == x.shape:
                pieces.append(_vtk_float_array(name, a.ravel(order="F"), n_comp=1))
            elif a.shape == x.shape + (3,):
                v = np.stack(
                    [a[..., 0].ravel(order="F"), a[..., 1].ravel(order="F"), a[..., 2].ravel(order="F")], axis=-1
                )
                pieces.append(_vtk_float_array(name, v, n_comp=3))
            else:
                raise ValueError(f"PointData {name!r} has shape {a.shape}, expected {x.shape} or {x.shape+(3,)}")
        pd_xml = "<PointData>\n" + "".join(pieces) + "</PointData>\n"

    pts_xml = _vtk_float_array(None, pts, n_comp=3)

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        f'  <StructuredGrid WholeExtent="{extent}">\n'
        f'    <Piece Extent="{extent}">\n'
        f"{pd_xml}"
        "      <Points>\n"
        f"        {pts_xml}"
        "      </Points>\n"
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "</VTKFile>\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    return path


def write_vtp_polyline(
    path: str | Path,
    *,
    points: np.ndarray,
    point_data: Dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a VTK XML PolyData (``.vtp``) containing a single polyline."""
    path = _as_path(path)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {pts.shape}")
    n = int(pts.shape[0])
    if n < 2:
        raise ValueError("polyline requires at least 2 points")

    point_data = point_data or {}
    pd_xml = ""
    if point_data:
        pieces = []
        for name, arr in point_data.items():
            a = np.asarray(arr)
            if a.shape == (n,):
                pieces.append(_vtk_float_array(name, a, n_comp=1))
            elif a.shape == (n, 3):
                pieces.append(_vtk_float_array(name, a, n_comp=3))
            else:
                raise ValueError(f"PointData {name!r} has shape {a.shape}, expected {(n,)} or {(n,3)}")
        pd_xml = "<PointData>\n" + "".join(pieces) + "</PointData>\n"

    # Polyline connectivity for a single line of length n.
    connectivity = np.arange(n, dtype=int)
    offsets = np.asarray([n], dtype=int)

    conn_xml = _vtk_int_array("connectivity", connectivity)
    off_xml = _vtk_int_array("offsets", offsets)
    pts_xml = _vtk_float_array(None, pts, n_comp=3)

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        "  <PolyData>\n"
        f'    <Piece NumberOfPoints="{n}" NumberOfVerts="0" NumberOfLines="1" NumberOfStrips="0" NumberOfPolys="0">\n'
        f"{pd_xml}"
        "      <Points>\n"
        f"        {pts_xml}"
        "      </Points>\n"
        "      <Lines>\n"
        f"        {conn_xml}"
        f"        {off_xml}"
        "      </Lines>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    return path


def _resolve_wout_path(input_path: Path, wout_path: str | Path | None) -> Path:
    if wout_path is not None:
        return _as_path(wout_path)
    name = input_path.name
    if name.startswith("input."):
        case = name.split("input.", 1)[1]
    else:
        case = input_path.stem
    candidates = [
        input_path.parent / f"wout_{case}_reference.nc",
        input_path.parent / f"wout_{case}.nc",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "wout file not found. Pass --wout or place a matching wout_*.nc next to the input."
    )


def export_vtk_surface_and_fieldline(
    *,
    input_path: str | Path,
    wout_path: str | Path | None = None,
    outdir: str | Path = "vtk_out",
    s_index: int = -1,
    hi_res: bool = False,
    export_volume: bool = False,
) -> dict[str, Path]:
    """Export one surface + a fieldline trace to VTK for ParaView.

    This helper reads a VMEC `wout_*.nc` and writes:
    - `surface_b.vts`: structured grid with Bx/By/Bz/Bmag on a surface
    - `fieldline.vtp`: a single fieldline polyline on that surface

    If `export_volume=True`, a coarse volume grid `volume.vts` is also written
    (with Bmag only).
    """
    from dataclasses import replace

    from .config import load_config
    from .field import b2_from_bsup, b_cartesian_from_bsup, bsup_from_geom, lamscale_from_phips
    from .fieldlines import trace_fieldline_on_surface
    from .geom import eval_geom
    from .grids import make_angle_grid
    from .static import build_static
    from .wout import read_wout, state_from_wout

    input_path = _as_path(input_path)
    wout_path = _resolve_wout_path(input_path, wout_path)
    outdir = _as_path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, _indata = load_config(str(input_path))
    ntheta = int(cfg.ntheta)
    nzeta = int(cfg.nzeta)
    if hi_res:
        ntheta = max(ntheta * 2, 128)
        nzeta = max(nzeta * 2, 128)
    if ntheta != cfg.ntheta or nzeta != cfg.nzeta:
        cfg = replace(cfg, ntheta=ntheta, nzeta=nzeta)

    grid = make_angle_grid(ntheta, nzeta, cfg.nfp, endpoint=False)
    static = build_static(cfg, grid=grid)

    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    geom = eval_geom(state, static)
    lamscale = lamscale_from_phips(wout.phips, static.s)
    bsupu, bsupv = bsup_from_geom(
        geom,
        phipf=wout.phipf,
        chipf=wout.chipf,
        nfp=int(wout.nfp),
        signgs=int(wout.signgs),
        lamscale=lamscale,
        flux_is_internal=False,
    )

    b2 = b2_from_bsup(geom, bsupu, bsupv)
    bmag = np.sqrt(np.asarray(b2))
    bcart = np.asarray(b_cartesian_from_bsup(geom, bsupu, bsupv, zeta=static.grid.zeta, nfp=int(wout.nfp)))

    ns = int(wout.ns)
    s_idx = int(s_index)
    if s_idx < 0:
        s_idx = ns + s_idx
    if s_idx < 0 or s_idx >= ns:
        raise ValueError(f"s_index={s_index} out of range for ns={ns}")

    zeta = np.asarray(static.grid.zeta)
    phi = zeta / float(max(1, int(wout.nfp)))
    cosphi = np.cos(phi)[None, :]
    sinphi = np.sin(phi)[None, :]

    R = np.asarray(geom.R)[s_idx]
    Z = np.asarray(geom.Z)[s_idx]
    x = R * cosphi
    y = R * sinphi
    z = Z

    bsurf = bcart[s_idx]
    bmag_s = bmag[s_idx]

    surface_path = outdir / "surface_b.vts"
    write_vts_structured_grid(
        surface_path,
        x=x,
        y=y,
        z=z,
        point_data={
            "Bx": bsurf[..., 0],
            "By": bsurf[..., 1],
            "Bz": bsurf[..., 2],
            "Bmag": bmag_s,
        },
    )

    n_steps = 2000 if hi_res else 800
    dphi = 2.0 * np.pi / float(n_steps - 1)
    fieldline = trace_fieldline_on_surface(
        R=R,
        Z=Z,
        bsupu=np.asarray(bsupu)[s_idx],
        bsupv=np.asarray(bsupv)[s_idx],
        Bmag=bmag_s,
        nfp=int(wout.nfp),
        theta0=0.0,
        phi0=0.0,
        n_steps=n_steps,
        dphi=dphi,
    )
    fieldline_path = outdir / "fieldline.vtp"
    write_vtp_polyline(
        fieldline_path,
        points=np.stack([fieldline.x, fieldline.y, fieldline.z], axis=-1),
        point_data={"Bmag": fieldline.Bmag},
    )

    paths = {"surface": surface_path, "fieldline": fieldline_path}

    if export_volume:
        Rvol = np.asarray(geom.R)
        Zvol = np.asarray(geom.Z)
        cosphi3 = np.cos(phi)[None, None, :]
        sinphi3 = np.sin(phi)[None, None, :]
        xvol = Rvol * cosphi3
        yvol = Rvol * sinphi3
        zvol = Zvol
        volume_path = outdir / "volume.vts"
        write_vts_structured_grid(
            volume_path,
            x=xvol,
            y=yvol,
            z=zvol,
            point_data={"Bmag": bmag},
        )
        paths["volume"] = volume_path

    return paths
