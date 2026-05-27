from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import vmec_jax.energy as energy_mod
from vmec_jax.energy import (
    _as_float_list,
    _has_nonzero_profile_coeffs,
    _iotaf_from_iotas,
    _make_torflux_jit,
    _poly_no_const,
    _poly_no_const_deriv,
    FluxProfiles,
    flux_profiles_from_indata,
    flux_profiles_from_indata_host_default,
    integrate_volume_density,
    magnetic_wb_from_state,
)
from vmec_jax.namelist import InData


def test_polynomial_and_iotaf_edge_branches():
    x = np.asarray([0.0, 0.5, 1.0])
    assert _as_float_list(None) == []
    assert _as_float_list([1, "2.5"]) == [1.0, 2.5]
    assert _as_float_list("3.5") == [3.5]

    np.testing.assert_allclose(np.asarray(_poly_no_const([], x)), np.zeros_like(x))
    np.testing.assert_allclose(np.asarray(_poly_no_const_deriv([], x)), np.zeros_like(x))
    np.testing.assert_allclose(np.asarray(_poly_no_const([2.0, 3.0], x)), 2.0 * x + 3.0 * x**2)
    np.testing.assert_allclose(np.asarray(_poly_no_const_deriv([2.0, 3.0], x)), 2.0 + 6.0 * x)

    np.testing.assert_allclose(np.asarray(_iotaf_from_iotas(np.asarray([0.2]), lrfp=False)), [0.2])
    np.testing.assert_allclose(np.asarray(_iotaf_from_iotas(np.asarray([0.0, 0.3]), lrfp=False)), [0.3, 0.3])
    np.testing.assert_allclose(
        np.asarray(_iotaf_from_iotas(np.asarray([0.0, 0.2, 0.4, 0.8]), lrfp=False)),
        [0.1, 0.3, 0.6, 1.0],
    )
    np.testing.assert_allclose(
        np.asarray(_iotaf_from_iotas(np.asarray([0.0, 0.2, 0.4, 0.8]), lrfp=True)),
        [1.0 / (1.5 / 0.2 - 0.5 / 0.4), 2.0 / (1.0 / 0.2 + 1.0 / 0.4), 2.0 / (1.0 / 0.4 + 1.0 / 0.8), 1.0 / (1.5 / 0.8 - 0.5 / 0.4)],
    )
    np.testing.assert_allclose(
        np.asarray(_iotaf_from_iotas(np.asarray([0.0, 0.0, 0.0]), lrfp=True)),
        [0.0, 0.0, 0.0],
    )


def test_torflux_cache_and_flux_profiles_nonrfp_and_rfp():
    deriv, torflux = _make_torflux_jit((1.0, 0.5), False, 123)
    deriv_again, torflux_again = _make_torflux_jit((1.0, 0.5), False, 123)
    assert deriv is deriv_again
    assert torflux is torflux_again
    np.testing.assert_allclose(np.asarray(deriv(np.asarray([0.0, 1.0]))), [1.0, 2.0])
    assert _make_torflux_jit((1.0,), True, 123) == (None, None)

    s = np.asarray([0.0, 0.5, 1.0])
    indata = InData(
        scalars={
            "PHIEDGE": 2.0,
            "APHI": [1.0],
            "AI": [0.25],
            "AC": [0.5],
        },
        indexed={},
    )
    flux = flux_profiles_from_indata(indata, s, signgs=-1)
    assert flux.signgs == -1
    assert np.asarray(flux.phipf).shape == (3,)
    assert np.asarray(flux.chipf).shape == (3,)
    assert np.asarray(flux.phips)[0] == 0.0
    assert float(np.asarray(flux.phipf[-1])) < 0.0

    rfp = InData(
        scalars={
            "PHIEDGE": 1.0,
            "LRFP": True,
            "AI": [2.0],
        },
        indexed={},
    )
    rfp_flux = flux_profiles_from_indata(rfp, s, signgs=1)
    np.testing.assert_allclose(np.asarray(rfp_flux.chipf), np.full(3, 1.0 / (4.0 * np.pi)))
    assert np.all(np.asarray(rfp_flux.phipf) > 0.0)
    assert np.asarray(rfp_flux.phips)[0] == 0.0


def test_flux_profiles_default_torflux_skips_unneeded_profile_evaluation(monkeypatch):
    s = np.asarray([0.0, 0.25, 0.5, 1.0])
    current_driven = InData(
        scalars={
            "PHIEDGE": 2.0,
            "APHI": [1.0],
            "AI": [],
            "AC": [1.0],
            "PCURR_TYPE": "cubic_spline_ip",
            "AC_AUX_S": [0.0, 1.0],
            "AC_AUX_F": [0.0, 2.0],
        },
        indexed={},
    )

    def fail_eval_profiles(_indata, _s):
        raise AssertionError("default torflux with no AI should not evaluate profiles")

    monkeypatch.setattr(energy_mod, "eval_profiles", fail_eval_profiles)

    flux = flux_profiles_from_indata(current_driven, s, signgs=1)

    np.testing.assert_allclose(np.asarray(flux.phipf), np.full_like(s, 1.0 / np.pi))
    np.testing.assert_allclose(np.asarray(flux.chipf), np.zeros_like(s))
    assert np.asarray(flux.phips)[0] == 0.0


def test_flux_profiles_default_torflux_uses_iota_profile_when_present(monkeypatch):
    s = np.asarray([0.0, 0.5, 1.0])
    iota_driven = InData(
        scalars={
            "PHIEDGE": 2.0,
            "APHI": [1.0],
            "AI": [0.25],
            "AC": [],
        },
        indexed={},
    )
    calls = {"n": 0}

    def fake_eval_profiles(_indata, s_grid):
        calls["n"] += 1
        return {"iota": np.full_like(np.asarray(s_grid), 0.25, dtype=float)}

    monkeypatch.setattr(energy_mod, "eval_profiles", fake_eval_profiles)

    flux = flux_profiles_from_indata(iota_driven, s, signgs=1)

    assert calls["n"] == 1
    np.testing.assert_allclose(np.asarray(flux.chipf), np.full_like(s, 0.25 / np.pi))


def test_profile_coeff_detection_treats_unparseable_values_as_nonzero():
    class BrokenCoeffs:
        def __float__(self):
            raise RuntimeError("not a scalar coefficient")

    assert _has_nonzero_profile_coeffs(BrokenCoeffs()) is True
    assert _has_nonzero_profile_coeffs([]) is False
    assert _has_nonzero_profile_coeffs([0.0, 1.0e-30]) is True


def test_flux_profiles_nondefault_aphi_without_iota_covers_zero_polflux_and_single_surface():
    s = np.asarray([0.5])
    indata = InData(scalars={"PHIEDGE": 1.0, "APHI": [1.0, 0.25], "AI": []}, indexed={})

    flux = flux_profiles_from_indata(indata, s, signgs=1)

    assert flux.signgs == 1
    np.testing.assert_allclose(np.asarray(flux.chipf), np.zeros_like(s))
    assert np.asarray(flux.phips).shape == (1,)
    assert np.asarray(flux.phips)[0] == 0.0


def test_flux_profiles_host_default_rejects_nondefault_cases_and_matches_default_meshes():
    s = np.asarray([0.0, 0.5, 1.0])
    default = InData(scalars={"PHIEDGE": 2.0, "APHI": [1.0], "AI": []}, indexed={})

    flux = flux_profiles_from_indata_host_default(default, s, signgs=-1)

    assert flux is not None
    np.testing.assert_allclose(flux.phipf, -np.ones_like(s) / np.pi)
    np.testing.assert_allclose(flux.chipf, np.zeros_like(s))
    np.testing.assert_allclose(flux.phips, [0.0, -1.0 / np.pi, -1.0 / np.pi])
    assert flux.lamscale > 0.0

    single = flux_profiles_from_indata_host_default(default, np.asarray([0.0]), signgs=1)
    assert single is not None
    np.testing.assert_allclose(single.phips, [0.0])
    assert float(single.lamscale) == 1.0

    with_iota = InData(scalars={"PHIEDGE": 2.0, "APHI": [1.0], "AI": [0.25, 0.5]}, indexed={})
    host_iota_flux = flux_profiles_from_indata_host_default(with_iota, s, signgs=1)
    full_iota_flux = flux_profiles_from_indata(with_iota, s, signgs=1)
    assert host_iota_flux is not None
    np.testing.assert_allclose(host_iota_flux.phipf, np.asarray(full_iota_flux.phipf))
    np.testing.assert_allclose(host_iota_flux.chipf, np.asarray(full_iota_flux.chipf))
    np.testing.assert_allclose(host_iota_flux.phips, np.asarray(full_iota_flux.phips))
    np.testing.assert_allclose(host_iota_flux.lamscale, np.asarray(full_iota_flux.lamscale))

    assert flux_profiles_from_indata_host_default(InData(scalars={"LRFP": True}, indexed={}), s, signgs=1) is None
    assert flux_profiles_from_indata_host_default(InData(scalars={"APHI": [1.0, 0.5]}, indexed={}), s, signgs=1) is None
    assert flux_profiles_from_indata_host_default(default, np.zeros((1, 2)), signgs=1) is None


def test_magnetic_wb_from_state_uses_vmec_energy_normalization(monkeypatch):
    static = SimpleNamespace(
        s=np.asarray([0.0, 1.0]),
        cfg=SimpleNamespace(nfp=1),
        grid=SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0, np.pi])),
    )
    geom = SimpleNamespace(sqrtg=np.ones((2, 2, 2)))
    flux = FluxProfiles(
        phipf=np.ones(2),
        chipf=np.zeros(2),
        phips=np.asarray([0.0, 1.0]),
        signgs=1,
        lamscale=np.asarray(1.0),
    )

    monkeypatch.setattr(energy_mod, "eval_geom", lambda state, static_arg: geom)
    monkeypatch.setattr(energy_mod, "flux_profiles_from_indata", lambda indata, s, *, signgs: flux)
    monkeypatch.setattr(energy_mod, "bsup_from_geom", lambda *args, **kwargs: (np.ones((2, 2, 2)), np.ones((2, 2, 2))))
    monkeypatch.setattr(energy_mod, "b2_from_bsup", lambda g, u, v: np.full((2, 2, 2), 4.0))

    wb, diag = magnetic_wb_from_state(object(), static, InData(scalars={}, indexed={}), signgs=1)

    expected_energy = np.sum(0.5 * np.full((2, 2, 2), 4.0)) * 1.0 * np.pi * np.pi
    np.testing.assert_allclose(np.asarray(diag["energy_total"]), expected_energy)
    np.testing.assert_allclose(np.asarray(wb), expected_energy / ((2.0 * np.pi) ** 2))
    assert diag["lamscale"] == flux.lamscale


def test_integrate_volume_density_single_surface_and_sign():
    density = np.ones((1, 2, 3))
    sqrtg = np.full((1, 2, 3), 2.0)
    s = np.asarray([0.0])
    theta = np.asarray([0.0, np.pi])
    zeta = np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])

    val = integrate_volume_density(density, sqrtg, s, theta, zeta, nfp=2, signgs=-1)

    expected = -np.sum(density * sqrtg) * (2.0 * np.pi / 2) * ((2.0 * np.pi / 3.0) / 2) * 2
    np.testing.assert_allclose(np.asarray(val), expected)
