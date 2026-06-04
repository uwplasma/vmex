from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from vmec_jax.visualization import (
    _as_path,
    _resolve_wout_path,
    _vtk_float_array,
    _vtk_int_array,
    write_vtp_polyline,
    write_vts_structured_grid,
    export_vtk_surface_and_fieldline,
)


def test_vts_writer_smoke(tmp_path):
    nx, ny = 2, 3
    x = np.arange(nx * ny, dtype=float).reshape(nx, ny)
    y = 2.0 * x
    z = -x
    v = np.stack([x, y, z], axis=-1)

    out = write_vts_structured_grid(
        tmp_path / "surf.vts",
        x=x,
        y=y,
        z=z,
        point_data={"scalar": x, "vec": v},
    )
    text = out.read_text()
    assert '<VTKFile type="StructuredGrid"' in text
    assert 'WholeExtent="0 1 0 2 0 0"' in text
    assert 'Name="scalar"' in text
    assert 'Name="vec"' in text


def test_vtp_polyline_writer_smoke(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    out = write_vtp_polyline(tmp_path / "line.vtp", points=pts, point_data={"s": np.array([0.0, 1.0, 2.0])})
    text = out.read_text()
    assert '<VTKFile type="PolyData"' in text
    assert 'NumberOfPoints="3"' in text
    assert 'Name="connectivity"' in text
    assert 'Name="offsets"' in text
    assert 'Name="s"' in text


def test_vtk_array_helpers_and_path_resolution(tmp_path):
    assert _as_path(tmp_path) is tmp_path
    assert _as_path(str(tmp_path / "x")).name == "x"

    scalar = _vtk_float_array("scalar", np.asarray([[1.0, 2.0]]), n_comp=1)
    vector = _vtk_float_array(None, np.asarray([[1.0, 2.0, 3.0]]), n_comp=3)
    ints = _vtk_int_array("ids", np.asarray([1.9, 2.1]))

    assert 'Name="scalar"' in scalar
    assert "NumberOfComponents=\"3\"" in vector
    assert "1 2" in ints

    input_dot = tmp_path / "input.case"
    input_dot.write_text("&INDATA\n/\n")
    reference = tmp_path / "wout_case_reference.nc"
    reference.write_text("reference")
    assert _resolve_wout_path(input_dot, None) == reference

    explicit = tmp_path / "explicit.nc"
    assert _resolve_wout_path(input_dot, explicit) == explicit

    reference.unlink()
    ordinary = tmp_path / "wout_case.nc"
    ordinary.write_text("ordinary")
    assert _resolve_wout_path(input_dot, None) == ordinary

    ordinary.unlink()
    input_plain = tmp_path / "custom_input.txt"
    input_plain.write_text("&INDATA\n/\n")
    (tmp_path / "wout_custom_input.nc").write_text("ordinary")
    assert _resolve_wout_path(input_plain, None).name == "wout_custom_input.nc"

    with pytest.raises(FileNotFoundError, match="wout file not found"):
        _resolve_wout_path(tmp_path / "input.missing", None)


def test_vts_writer_validates_shapes_and_supports_3d_point_data(tmp_path):
    x = np.ones((2, 3, 2))
    y = np.ones_like(x) * 2.0
    z = np.ones_like(x) * 3.0
    vec = np.stack([x, y, z], axis=-1)

    out = write_vts_structured_grid(
        tmp_path / "volume.vts",
        x=x,
        y=y,
        z=z,
        point_data={"bvec": vec},
    )
    text = out.read_text()
    assert 'WholeExtent="0 1 0 2 0 1"' in text
    assert 'Name="bvec"' in text

    with pytest.raises(ValueError, match="identical shapes"):
        write_vts_structured_grid(tmp_path / "bad.vts", x=np.ones((2, 2)), y=np.ones((2, 3)), z=np.ones((2, 2)))
    with pytest.raises(ValueError, match="2D or 3D"):
        write_vts_structured_grid(tmp_path / "bad.vts", x=np.ones((2,)), y=np.ones((2,)), z=np.ones((2,)))
    with pytest.raises(ValueError, match="PointData"):
        write_vts_structured_grid(
            tmp_path / "bad.vts",
            x=np.ones((2, 2)),
            y=np.ones((2, 2)),
            z=np.ones((2, 2)),
            point_data={"bad": np.ones((3,))},
        )


def test_vtp_polyline_writer_validates_shapes_and_vector_data(tmp_path):
    pts = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vec = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    out = write_vtp_polyline(tmp_path / "line_vec.vtp", points=pts, point_data={"B": vec})
    assert 'Name="B"' in out.read_text()

    with pytest.raises(ValueError, match="points must be"):
        write_vtp_polyline(tmp_path / "bad.vtp", points=np.ones((2, 2)))
    with pytest.raises(ValueError, match="at least 2"):
        write_vtp_polyline(tmp_path / "bad.vtp", points=np.ones((1, 3)))
    with pytest.raises(ValueError, match="PointData"):
        write_vtp_polyline(tmp_path / "bad.vtp", points=pts, point_data={"bad": np.ones((3,))})


def test_export_vtk_surface_fieldline_and_volume_with_synthetic_kernels(monkeypatch, tmp_path):
    import vmec_jax.config as config_module
    import vmec_jax.field as field_module
    import vmec_jax.fieldlines as fieldlines_module
    import vmec_jax.geom as geom_module
    import vmec_jax.static as static_module
    import vmec_jax.wout as wout_module

    input_path = tmp_path / "input.case"
    wout_path = tmp_path / "wout_case.nc"
    input_path.write_text("&INDATA\n/\n")
    wout_path.write_text("placeholder")

    @dataclass(frozen=True)
    class Cfg:
        ntheta: int = 2
        nzeta: int = 3
        nfp: int = 1

    cfg = Cfg()
    static_holder = {}

    def fake_build_static(cfg_arg, *, grid):
        static = type("Static", (), {"cfg": cfg_arg, "grid": grid, "s": np.asarray([0.0, 1.0])})()
        static_holder["static"] = static
        return static

    wout = type(
        "Wout",
        (),
        {
            "ns": 2,
            "nfp": 1,
            "signgs": 1,
            "phips": np.asarray([0.0, 1.0]),
            "phipf": np.asarray([1.0, 1.0]),
            "chipf": np.asarray([0.0, 0.0]),
        },
    )()
    state = object()
    def fake_eval_geom(_state, static):
        shape = (2, int(static.grid.ntheta), int(static.grid.nzeta))
        return type(
            "Geom",
            (),
            {
                "R": np.ones(shape),
                "Z": np.zeros(shape),
            },
        )()

    def fake_bsup_from_geom(geom_arg, *_args, **_kwargs):
        shape = np.asarray(geom_arg.R).shape
        return np.ones(shape) * 0.2, np.ones(shape) * 0.3

    def fake_bcart_from_bsup(geom_arg, *_args, **_kwargs):
        bcart = np.zeros(np.asarray(geom_arg.R).shape + (3,))
        bcart[..., 0] = 1.0
        bcart[..., 1] = 2.0
        bcart[..., 2] = 3.0
        return bcart

    monkeypatch.setattr(config_module, "load_config", lambda _path: (cfg, object()))
    monkeypatch.setattr(static_module, "build_static", fake_build_static)
    monkeypatch.setattr(wout_module, "read_wout", lambda _path: wout)
    monkeypatch.setattr(wout_module, "state_from_wout", lambda _wout: state)
    monkeypatch.setattr(geom_module, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(field_module, "lamscale_from_phips", lambda _phips, _s: 1.0)
    monkeypatch.setattr(field_module, "bsup_from_geom", fake_bsup_from_geom)
    monkeypatch.setattr(field_module, "b2_from_bsup", lambda geom_arg, *_args, **_kwargs: np.ones(np.asarray(geom_arg.R).shape) * 4.0)
    monkeypatch.setattr(field_module, "b_cartesian_from_bsup", fake_bcart_from_bsup)
    monkeypatch.setattr(
        fieldlines_module,
        "trace_fieldline_on_surface",
        lambda **_kwargs: type(
            "Line",
            (),
            {
                "x": np.asarray([0.0, 1.0, 2.0]),
                "y": np.asarray([0.0, 0.5, 1.0]),
                "z": np.asarray([0.0, 0.0, 0.0]),
                "Bmag": np.asarray([2.0, 2.0, 2.0]),
            },
        )(),
    )

    paths = export_vtk_surface_and_fieldline(
        input_path=input_path,
        wout_path=wout_path,
        outdir=tmp_path / "vtk",
        export_volume=True,
    )

    assert set(paths) == {"surface", "fieldline", "volume"}
    assert paths["surface"].read_text().count('Name="Bmag"') == 1
    assert 'Name="Bmag"' in paths["fieldline"].read_text()
    assert 'WholeExtent="0 1 0 1 0 2"' in paths["volume"].read_text()
    assert static_holder["static"].grid.theta.shape[0] == 2

    hi_res_paths = export_vtk_surface_and_fieldline(
        input_path=input_path,
        wout_path=wout_path,
        outdir=tmp_path / "vtk_hi_res",
        hi_res=True,
    )
    assert set(hi_res_paths) == {"surface", "fieldline"}
    assert static_holder["static"].grid.theta.shape[0] == 128
    assert static_holder["static"].grid.zeta.shape[0] == 128

    with pytest.raises(ValueError, match="out of range"):
        export_vtk_surface_and_fieldline(
            input_path=input_path,
            wout_path=wout_path,
            outdir=tmp_path / "vtk_bad",
            s_index=99,
        )
