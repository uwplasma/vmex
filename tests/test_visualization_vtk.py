from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.visualization import (
    _as_path,
    _resolve_wout_path,
    _vtk_float_array,
    _vtk_int_array,
    write_vtp_polyline,
    write_vts_structured_grid,
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
