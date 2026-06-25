from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.io.wout.netcdf import (
    NYQUIST_FOURIER_FIELD_NAMES,
    read_mode_table,
    read_nyquist_fourier_fields,
    read_optional_int_scalar,
    read_type_field,
    write_fixed_width_string_variable,
    write_float_variable,
    write_int_variable,
    write_nyquist_fourier_fields,
)


class _FakeReadVar:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return self.value


class _FakeWriteVar:
    def __init__(self, dtype: str, dims: tuple[str, ...]):
        self.dtype = dtype
        self.dims = dims
        self.value = None

    def __setitem__(self, key, value) -> None:
        self.value = np.asarray(value)


class _FakeDataset:
    def __init__(self):
        self.variables = {}

    def createVariable(self, name: str, dtype: str, dims: tuple[str, ...]) -> _FakeWriteVar:
        var = _FakeWriteVar(dtype, dims)
        self.variables[name] = var
        return var


def test_read_mode_table_fills_partial_masks_and_rejects_fully_masked() -> None:
    variables = {
        "xm": _FakeReadVar(np.ma.array([0.0, 99.0, 2.0], mask=[False, True, False])),
        "xn": _FakeReadVar(np.ma.array([1.0, 2.0], mask=[True, True])),
    }

    np.testing.assert_array_equal(read_mode_table(variables, "xm", path=Path("wout.nc")), [0, 0, 2])
    with pytest.raises(ValueError, match=r"Incomplete or masked wout mode metadata \(xn\)"):
        read_mode_table(variables, "xn", path=Path("wout.nc"))


def test_read_optional_int_scalar_uses_schema_scalar_defaults() -> None:
    variables = {
        "present": _FakeReadVar(np.asarray([4.7])),
        "masked": _FakeReadVar(np.ma.array([99], mask=[True])),
    }

    assert read_optional_int_scalar(variables, "present", 0) == 4
    assert read_optional_int_scalar(variables, "masked", 8) == 8
    assert read_optional_int_scalar(variables, "missing", 9) == 9


def test_read_type_field_decodes_fixed_width_character_arrays() -> None:
    variables = {
        "bytes": _FakeReadVar(np.asarray(list("power_series   "), dtype="S1")),
        "unicode": _FakeReadVar(np.asarray(list("akima_spline  "), dtype="U1")),
        "unicode_scalar": _FakeReadVar(np.asarray("cubic_spline   ", dtype="U16")),
        "object_chars": _FakeReadVar(np.asarray(list("line_segment  "), dtype=object)),
        "numeric_fallback": _FakeReadVar(np.asarray([1, 2])),
    }

    assert read_type_field(variables, "bytes") == "power_series"
    assert read_type_field(variables, "unicode") == "akima_spline"
    assert read_type_field(variables, "unicode_scalar") == "cubic_spline"
    assert read_type_field(variables, "object_chars") == "line_segment"
    assert read_type_field(variables, "numeric_fallback") == "[1 2]"
    assert read_type_field(variables, "missing") == ""


def test_read_nyquist_fourier_fields_uses_same_shape_defaults() -> None:
    gmnc = np.arange(6.0).reshape(2, 3)
    bsupumnc = np.arange(8.0).reshape(2, 4) + 10.0
    bsupvmnc = np.arange(10.0).reshape(2, 5) + 20.0
    variables = {
        "gmnc": _FakeReadVar(gmnc),
        "bsupumnc": _FakeReadVar(bsupumnc),
        "bsupvmnc": _FakeReadVar(bsupvmnc),
        "bmns": _FakeReadVar(gmnc + 30.0),
    }

    fields = read_nyquist_fourier_fields(variables)

    np.testing.assert_array_equal(fields["gmnc"], gmnc)
    np.testing.assert_array_equal(fields["bsupumnc"], bsupumnc)
    np.testing.assert_array_equal(fields["bsupvmnc"], bsupvmnc)
    assert tuple(fields) == NYQUIST_FOURIER_FIELD_NAMES
    np.testing.assert_array_equal(fields["gmns"], np.zeros_like(gmnc))
    np.testing.assert_array_equal(fields["bsupumns"], np.zeros_like(bsupumnc))
    np.testing.assert_array_equal(fields["bsubumnc"], np.zeros_like(bsupumnc))
    np.testing.assert_array_equal(fields["bsubvmns"], np.zeros_like(bsupvmnc))
    np.testing.assert_array_equal(fields["bsubsmns"], np.zeros_like(bsupvmnc))
    np.testing.assert_array_equal(fields["bmnc"], np.zeros_like(gmnc))
    np.testing.assert_array_equal(fields["bmns"], gmnc + 30.0)


def test_write_helpers_create_expected_netcdf_dtypes_and_fixed_width_strings() -> None:
    ds = _FakeDataset()

    write_int_variable(ds, "ns", (), 3)
    write_float_variable(ds, "xm", ("mn_mode",), np.asarray([0, 1]))
    write_fixed_width_string_variable(ds, "pcurr_type", "power_series_long_name", width=12, dim="dim_00012")

    assert ds.variables["ns"].dtype == "i4"
    assert ds.variables["ns"].dims == ()
    assert ds.variables["ns"].value.dtype == np.dtype("int32")
    assert ds.variables["xm"].dtype == "f8"
    assert ds.variables["xm"].dims == ("mn_mode",)
    assert ds.variables["xm"].value.dtype == np.dtype("float64")
    assert ds.variables["pcurr_type"].dtype == "S1"
    assert ds.variables["pcurr_type"].dims == ("dim_00012",)
    assert b"".join(ds.variables["pcurr_type"].value).decode("utf-8") == "power_series"


def test_write_nyquist_fourier_fields_writes_expected_group() -> None:
    wout = SimpleNamespace(**{name: np.full((2, 3), i, dtype=float) for i, name in enumerate(NYQUIST_FOURIER_FIELD_NAMES)})
    ds = _FakeDataset()

    write_nyquist_fourier_fields(ds, wout)

    assert tuple(ds.variables) == NYQUIST_FOURIER_FIELD_NAMES
    for i, name in enumerate(NYQUIST_FOURIER_FIELD_NAMES):
        assert ds.variables[name].dtype == "f8"
        assert ds.variables[name].dims == ("radius", "mn_mode_nyq")
        np.testing.assert_array_equal(ds.variables[name].value, np.full((2, 3), i, dtype=float))
