from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import has_jax, jax, jnp
from vmec_jax.namelist import InData
from vmec_jax.io.wout_files.netcdf import (
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
from vmec_jax.wout import _chipf_from_chips, _icurv_full_mesh_from_indata, read_wout, write_wout


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


def test_icurv_full_mesh_is_jit_safe_for_current_driven_profile():
    if not has_jax():
        return

    indata = InData(
        scalars={
            "NCURR": 1,
            "CURTOR": 2.0,
            "PCURR_TYPE": "power_series",
            "AC": [1.0, 0.0],
        },
        indexed={},
    )
    s = jnp.linspace(0.0, 1.0, 5)

    @jax.jit
    def _eval(s_grid):
        return _icurv_full_mesh_from_indata(indata=indata, s_full=s_grid, signgs=-1)

    out = np.asarray(_eval(s))

    assert out.shape == (5,)
    assert out[0] == 0.0
    assert np.all(np.isfinite(out))
    assert np.any(np.abs(out[1:]) > 0.0)


@pytest.mark.full
def test_write_wout_is_vmecplot2_compatible(tmp_path: Path) -> None:
    pytest.importorskip("netCDF4")
    pytest.importorskip("scipy")

    ref = Path(__file__).resolve().parents[3] / "examples" / "data" / "wout_circular_tokamak_reference.nc"
    if not ref.exists():
        pytest.skip("Reference wout not found")

    wout = read_wout(ref)
    out = tmp_path / "wout_test.nc"
    write_wout(out, wout, overwrite=True)

    from scipy.io import netcdf

    f = netcdf.netcdf_file(out, "r", mmap=False)
    try:
        required = [
            "phi",
            "iotaf",
            "presf",
            "iotas",
            "pres",
            "ns",
            "nfp",
            "xn",
            "xm",
            "xn_nyq",
            "xm_nyq",
            "rmnc",
            "zmns",
            "bmnc",
            "raxis_cc",
            "zaxis_cs",
            "buco",
            "bvco",
            "jcuru",
            "jcurv",
            "lasym__logical__",
            "ac_aux_s",
            "ac_aux_f",
            "pcurr_type",
            "Aminor_p",
            "Rmajor_p",
            "aspect",
            "betatotal",
            "betapol",
            "betator",
            "betaxis",
            "ctor",
            "DMerc",
        ]
        for name in required:
            assert name in f.variables
    finally:
        f.close()


def test_write_wout_mode_tables_use_float_storage(tmp_path: Path) -> None:
    netCDF4 = pytest.importorskip("netCDF4")

    ref = Path(__file__).resolve().parents[3] / "examples" / "data" / "wout_circular_tokamak.nc"
    if not ref.exists():
        pytest.skip("Reference wout not found")

    wout = read_wout(ref)
    out = tmp_path / "wout_mode_dtype.nc"
    write_wout(out, wout, overwrite=True)

    with netCDF4.Dataset(out) as ds:
        for name in ("xm", "xn", "xm_nyq", "xn_nyq"):
            assert ds.variables[name].dtype == np.dtype("float64")
        for name in ("mnmax", "mnmax_nyq", "mpol_nyq", "ntor_nyq"):
            assert ds.variables[name].dtype == np.dtype("int32")


def test_chipf_from_chips_is_jittable():
    chips = jnp.asarray([0.0, 1.0, 2.0, 3.0], dtype=jnp.float64)
    chipf = jax.jit(_chipf_from_chips)(chips)
    np.testing.assert_allclose(np.asarray(chipf), np.array([0.5, 1.5, 2.5, 3.5]))
