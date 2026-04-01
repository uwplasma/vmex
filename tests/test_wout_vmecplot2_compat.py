from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp


from vmec_jax.wout import read_wout, write_wout, _chipf_from_chips


@pytest.mark.full
def test_write_wout_is_vmecplot2_compatible(tmp_path: Path) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    scipy = pytest.importorskip("scipy")

    ref = Path(__file__).resolve().parents[1] / "examples" / "data" / "wout_circular_tokamak_reference.nc"
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

    ref = Path(__file__).resolve().parents[1] / "examples" / "data" / "wout_circular_tokamak.nc"
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
