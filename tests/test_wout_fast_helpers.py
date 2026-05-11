from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.namelist import InData
import vmec_jax.wout as wout_module
from vmec_jax.wout import (
    MU0,
    _bool_from_nc,
    _chipf_from_chips,
    _icurv_full_mesh_from_indata,
    _nc_scalar,
    _pshalf_from_s,
    _safe_divide,
    assert_main_modes_match_wout,
)
from vmec_jax.wout_schema import WoutData as SchemaWoutData
from vmec_jax.wout_schema import _bool_from_nc as schema_bool_from_nc
from vmec_jax.wout_schema import _nc_scalar as schema_nc_scalar
from vmec_jax.wout_schema import assert_main_modes_match_wout as schema_assert_main_modes_match_wout


def test_wout_half_mesh_and_flux_derivative_conventions() -> None:
    s_full = np.asarray([0.0, 0.25, 1.0])
    np.testing.assert_allclose(_pshalf_from_s(s_full), np.sqrt([0.125, 0.125, 0.625]))
    np.testing.assert_allclose(_pshalf_from_s(np.asarray([0.36])), [0.6])

    chips = np.asarray([0.0, 1.0, 4.0, 9.0])
    np.testing.assert_allclose(_chipf_from_chips(chips), [-0.5, 2.5, 6.5, 11.5])
    np.testing.assert_allclose(_chipf_from_chips(np.asarray([2.0, 5.0])), [5.0, 6.5])


def test_safe_divide_uses_unit_denominator_for_exact_zeros() -> None:
    num = np.asarray([2.0, 4.0, 6.0])
    den = np.asarray([1.0, 0.0, -2.0])
    np.testing.assert_allclose(_safe_divide(num, den), [2.0, 4.0, -3.0])


def test_current_profile_full_mesh_uses_vmec_half_mesh_normalization() -> None:
    s_full = np.asarray([0.0, 0.25, 1.0])
    indata = InData(
        scalars={
            "NCURR": 1,
            "CURTOR": 10.0,
            "PCURR_TYPE": "power_series",
            "AC": [2.0],
        },
        indexed={},
    )

    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s_full, signgs=-1)
    expected_scale = -MU0 * 10.0 / (2.0 * np.pi) / 2.0
    # I(s_half)=2*s_half and VMEC explicitly zeroes the axis value.
    np.testing.assert_allclose(np.asarray(icurv), expected_scale * np.asarray([0.0, 0.25, 1.25]))

    no_current = InData(scalars={"NCURR": 0, "CURTOR": 10.0, "AC": [2.0]}, indexed={})
    np.testing.assert_allclose(np.asarray(_icurv_full_mesh_from_indata(indata=no_current, s_full=s_full, signgs=1)), 0.0)

    zero_edge = InData(scalars={"NCURR": 1, "CURTOR": 10.0, "AC": [0.0]}, indexed={})
    np.testing.assert_allclose(np.asarray(_icurv_full_mesh_from_indata(indata=zero_edge, s_full=s_full, signgs=1)), 0.0)


def test_netcdf_scalar_helpers_handle_masked_and_fallback_values() -> None:
    assert _bool_from_nc(np.ma.array([1], mask=[False])) is True
    assert _bool_from_nc(np.ma.array([99], mask=[True])) is False
    assert _nc_scalar(np.ma.array([3.25], mask=[False])) == 3.25
    assert _nc_scalar(np.ma.array([3.25], mask=[False]), as_int=True) == 3
    assert _nc_scalar(object(), default=7.0) == 7.0
    assert _nc_scalar(object(), default=7.0, as_int=True) == 7


def test_wout_main_mode_order_contract_detects_mismatches() -> None:
    good = SimpleNamespace(
        path=Path("wout_good.nc"),
        mpol=2,
        ntor=1,
        nfp=3,
        xm=np.asarray([0, 0, 1, 1, 1]),
        xn=np.asarray([0, 3, -3, 0, 3]),
    )
    assert_main_modes_match_wout(wout=good)

    bad_m = SimpleNamespace(**{**good.__dict__, "xm": np.asarray([0, 1])})
    with pytest.raises(ValueError, match="Mode count mismatch"):
        assert_main_modes_match_wout(wout=bad_m)

    bad_order = SimpleNamespace(
        **{**good.__dict__, "xm": np.asarray([0, 0, 1, 1, 1]), "xn": np.asarray([0, -3, -3, 0, 3])}
    )
    with pytest.raises(ValueError, match="xn ordering"):
        assert_main_modes_match_wout(wout=bad_order)


def test_wout_schema_symbols_remain_reexported_from_wout() -> None:
    assert wout_module.WoutData is SchemaWoutData
    assert wout_module._bool_from_nc is schema_bool_from_nc
    assert wout_module._nc_scalar is schema_nc_scalar
    assert wout_module.assert_main_modes_match_wout is schema_assert_main_modes_match_wout
