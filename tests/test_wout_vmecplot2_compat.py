from pathlib import Path

import pytest

from vmec_jax.wout import read_wout, write_wout


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
