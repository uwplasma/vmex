from __future__ import annotations

import numpy as np
import pytest

from vmec_jax import profiles as profiles_mod
from vmec_jax._compat import has_jax, jax, jnp
from vmec_jax.config import load_config
from vmec_jax.namelist import InData
from vmec_jax.profiles import ProfileInputs, eval_profiles


def test_profile_private_helpers_and_fallbacks():
    assert profiles_mod._as_float_list(None) == []
    assert profiles_mod._as_float_list(3) == [3.0]
    assert profiles_mod._as_float_list(["2.5"]) == [2.5]
    assert profiles_mod._lower(["'Two_Power'"], "power_series") == "two_power"
    assert profiles_mod._lower([], "power_series") == "power_series"
    np.testing.assert_allclose(np.asarray(profiles_mod._coeff_array([], nmin=3)), np.zeros(3))

    indata = InData(
        scalars={
            "AC_AUX_S": [0.0, 0.4, 0.3, 1.0],
            "AC_AUX_F": [1.0, 2.0, 3.0, 4.0],
        },
        indexed={},
    )
    s_aux, f_aux = profiles_mod._aux_profile_arrays(indata, "AC")
    np.testing.assert_allclose(np.asarray(s_aux), [0.0, 0.4])
    np.testing.assert_allclose(np.asarray(f_aux), [1.0, 2.0])


def test_numpy_profile_helpers_cover_degenerate_and_unparseable_inputs(monkeypatch):
    class BrokenArray:
        def __array__(self, *_args, **_kwargs):
            raise RuntimeError("cannot materialize")

    np.testing.assert_array_equal(profiles_mod._profile_coeffs_np(BrokenArray()), np.asarray([], dtype=np.float64))

    x = np.asarray([0.0, 0.5, 1.0])
    np.testing.assert_allclose(profiles_mod._two_power_np([3.0], x), 3.0 * np.ones_like(x))
    np.testing.assert_allclose(profiles_mod._cubic_spline_profile_np([], [], x, integrate=False), np.zeros_like(x))
    np.testing.assert_allclose(profiles_mod._cubic_spline_profile_np([0.0], [2.0], x, integrate=False), np.full_like(x, 2.0))
    np.testing.assert_allclose(profiles_mod._cubic_spline_profile_np([0.0], [2.0], x, integrate=True), 2.0 * x)

    assert profiles_mod._can_use_numpy_profile_eval(BrokenArray()) is False

    if has_jax():
        arr = jnp.asarray([1.0, 2.0])
        assert profiles_mod._as_float_list(arr) is arr

        monkeypatch.setattr(profiles_mod, "_is_jax_tracer", lambda _value: True)
        assert profiles_mod._can_use_numpy_profile_eval(jnp.asarray([0.0, 1.0])) is False


def test_jax_and_numpy_tabulated_profile_helpers_match_for_line_segment_and_akima():
    if not has_jax():
        pytest.skip("JAX is required for differentiable tabulated profile coverage")

    x = np.asarray([0.0, 0.1, 0.35, 0.7, 1.0])
    line_knots = np.asarray([0.0, 0.4, 1.0])
    line_values = np.asarray([1.0, 2.0, 0.5])
    akima_knots = np.asarray([0.0, 0.15, 0.45, 0.8, 1.0])
    akima_values = np.asarray([0.0, 0.4, 0.1, 0.8, 0.6])

    for integrate in (False, True):
        line_jax = profiles_mod._line_segment_profile(
            jnp.asarray(line_knots),
            jnp.asarray(line_values),
            jnp.asarray(x),
            integrate=integrate,
        )
        line_np = profiles_mod._line_segment_profile_np(
            line_knots,
            line_values,
            x,
            integrate=integrate,
        )
        np.testing.assert_allclose(np.asarray(line_jax), line_np, rtol=1.0e-12, atol=1.0e-12)

        akima_jax = profiles_mod._akima_spline_profile(
            jnp.asarray(akima_knots),
            jnp.asarray(akima_values),
            jnp.asarray(x),
            integrate=integrate,
        )
        akima_np = profiles_mod._akima_spline_profile_np(
            akima_knots,
            akima_values,
            x,
            integrate=integrate,
        )
        np.testing.assert_allclose(np.asarray(akima_jax), akima_np, rtol=1.0e-12, atol=1.0e-12)

    # Fewer than four Akima knots intentionally fall back to the cubic helper.
    short_knots = np.asarray([0.0, 0.5, 1.0])
    short_values = np.asarray([0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(
            profiles_mod._akima_spline_profile(
                jnp.asarray(short_knots),
                jnp.asarray(short_values),
                jnp.asarray(x),
                integrate=False,
            )
        ),
        profiles_mod._akima_spline_profile_np(short_knots, short_values, x, integrate=False),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        np.asarray(profiles_mod._line_segment_profile(jnp.asarray([]), jnp.asarray([]), jnp.asarray(x), integrate=False)),
        np.zeros_like(x),
    )
    np.testing.assert_allclose(
        np.asarray(
            profiles_mod._line_segment_profile(
                jnp.asarray([0.0]),
                jnp.asarray([2.0]),
                jnp.asarray(x),
                integrate=True,
            )
        ),
        2.0 * x,
    )


def test_two_power_and_empty_current_profile_branches():
    indata = InData(
        scalars={
            "PMASS_TYPE": "two_power",
            "PIOTA_TYPE": "power_series",
            "PCURR_TYPE": "cubic_spline_i",
            "AM": [2.0, 2.0, 1.0],
            "AI": [0.0, 2.0],
            "AC": [1.0],
            "AC_AUX_S": [],
            "AC_AUX_F": [],
            "PRES_SCALE": 1.0,
            "BLOAT": 1.0,
            "SPRES_PED": 0.5,
            "LRFP": True,
            "NCURR": 1,
        },
        indexed={},
    )
    s = np.asarray([0.0, 0.25, 0.75])
    prof = eval_profiles(indata, s)
    assert np.isinf(np.asarray(prof["iota"])[0])
    np.testing.assert_allclose(np.asarray(prof["iota"])[1:], [2.0, 2.0 / 3.0])
    np.testing.assert_allclose(np.asarray(prof["current"]), np.zeros_like(s))


def test_current_profile_spline_i_and_ip_follow_vmec_integral_convention():
    s = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    base_scalars = {
        "PMASS_TYPE": "power_series",
        "PIOTA_TYPE": "power_series",
        "AM": [0.0],
        "AI": [],
        "AC": [1.0],
        "AC_AUX_S": [0.0, 1.0],
        "AC_AUX_F": [0.0, 2.0],
        "PRES_SCALE": 1.0,
        "BLOAT": 1.0,
        "SPRES_PED": 1.0,
        "NCURR": 1,
    }

    integrated = eval_profiles(
        InData(scalars={**base_scalars, "PCURR_TYPE": "cubic_spline_ip"}, indexed={}),
        s,
    )
    direct = eval_profiles(
        InData(scalars={**base_scalars, "PCURR_TYPE": "cubic_spline_i"}, indexed={}),
        s,
    )

    # VMEC's cubic_spline_ip parameterizes I'(s); with I'(s)=2s the enclosed
    # current is exactly I(s)=s^2. cubic_spline_i stores I(s) directly.
    np.testing.assert_allclose(np.asarray(integrated["current"]), s * s, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(direct["current"]), 2.0 * s, rtol=1.0e-12, atol=1.0e-12)


def test_pressure_and_iota_cubic_spline_profiles_follow_vmec_knots():
    knots = [0.0, 0.25, 0.5, 0.75, 1.0]
    pressure_values = [1.0 + 2.0 * s + 3.0 * s * s for s in knots]
    iota_values = [0.3 + 0.2 * s - 0.1 * s * s for s in knots]
    indata = InData(
        scalars={
            "PMASS_TYPE": "cubic_spline",
            "PIOTA_TYPE": "cubic_spline",
            "PCURR_TYPE": "line_segment_ip",
            "AM": [0.0],
            "AI": [0.0],
            "AC": [1.0],
            "AM_AUX_S": knots,
            "AM_AUX_F": pressure_values,
            "AI_AUX_S": knots,
            "AI_AUX_F": iota_values,
            "AC_AUX_S": [0.0, 0.5, 1.0],
            "AC_AUX_F": [2.0, 4.0, 2.0],
            "PRES_SCALE": 5.0,
            "BLOAT": 1.0,
            "SPRES_PED": 1.0,
            "LRFP": False,
            "NCURR": 0,
        },
        indexed={},
    )
    s = np.asarray([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    prof = eval_profiles(indata, s)

    np.testing.assert_allclose(
        np.asarray(prof["pressure_pa"]),
        5.0 * (1.0 + 2.0 * s + 3.0 * s * s),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(prof["iota"]),
        0.3 + 0.2 * s - 0.1 * s * s,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    # Piecewise-linear I'(s): 2+4s on [0,0.5], 6-4s on [0.5,1].
    expected_current = np.where(s <= 0.5, 2.0 * s + 2.0 * s**2, 1.5 + 6.0 * (s - 0.5) - 2.0 * (s * s - 0.25))
    np.testing.assert_allclose(np.asarray(prof["current"]), expected_current, rtol=1.0e-12, atol=1.0e-12)


def test_akima_spline_profiles_cover_pressure_iota_and_current():
    knots = [0.0, 0.25, 0.5, 0.75, 1.0]
    indata = InData(
        scalars={
            "PMASS_TYPE": "akima_spline",
            "PIOTA_TYPE": "akima_spline",
            "PCURR_TYPE": "akima_spline_ip",
            "AM": [0.0],
            "AI": [0.0],
            "AC": [1.0],
            "AM_AUX_S": knots,
            "AM_AUX_F": [1.0 + 2.0 * s for s in knots],
            "AI_AUX_S": knots,
            "AI_AUX_F": [0.3 + 0.1 * s for s in knots],
            "AC_AUX_S": knots,
            "AC_AUX_F": [2.0 * s for s in knots],
            "PRES_SCALE": 3.0,
            "BLOAT": 1.0,
            "SPRES_PED": 1.0,
            "LRFP": False,
            "NCURR": 1,
        },
        indexed={},
    )
    s = np.asarray([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    prof = eval_profiles(indata, s)

    np.testing.assert_allclose(np.asarray(prof["pressure_pa"]), 3.0 * (1.0 + 2.0 * s), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(prof["iota"]), 0.3 + 0.1 * s, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(prof["current"]), s * s, rtol=1.0e-12, atol=1.0e-12)


def test_akima_spline_jax_and_numpy_paths_match_for_nonlinear_knots():
    if not has_jax():
        pytest.skip("JAX is required for concrete JAX array Akima profile coverage")

    knots = [0.0, 0.2, 0.5, 0.8, 1.0]
    indata = InData(
        scalars={
            "PMASS_TYPE": "akima_spline",
            "PIOTA_TYPE": "akima_spline",
            "PCURR_TYPE": "akima_spline_i",
            "AM": [0.0],
            "AI": [0.0],
            "AC": [1.0],
            "AM_AUX_S": knots,
            "AM_AUX_F": [1.0, 0.7, 0.4, 0.2, 0.0],
            "AI_AUX_S": knots,
            "AI_AUX_F": [0.3, 0.34, 0.42, 0.48, 0.52],
            "AC_AUX_S": knots,
            "AC_AUX_F": [0.0, 0.02, 0.1, 0.18, 0.2],
            "PRES_SCALE": 1.0,
            "BLOAT": 1.0,
            "SPRES_PED": 1.0,
            "LRFP": False,
            "NCURR": 1,
        },
        indexed={},
    )
    s = np.linspace(0.0, 1.0, 11)
    prof_np = eval_profiles(indata, s)
    prof_jax = eval_profiles(indata, jnp.asarray(s))
    for key in ("pressure_pa", "pressure", "iota", "current"):
        np.testing.assert_allclose(np.asarray(prof_jax[key]), np.asarray(prof_np[key]), rtol=1.0e-12, atol=1.0e-12)


def test_bundled_profile_spline_input_evaluates_pressure_and_iota():
    _cfg, indata = load_config("examples/data/input.profile_splines")
    s = np.linspace(0.0, 1.0, 6)
    prof = eval_profiles(indata, s)

    assert str(indata.get("PMASS_TYPE")).strip('"').lower() == "cubic_spline"
    assert str(indata.get("PIOTA_TYPE")).strip('"').lower() == "cubic_spline"
    assert np.all(np.asarray(prof["pressure_pa"]) >= 0.0)
    assert np.all(np.diff(np.asarray(prof["pressure_pa"])) <= 0.0)
    np.testing.assert_allclose(np.asarray(prof["iota"])[[0, -1]], [1.05, 0.70], rtol=1.0e-12)


def test_concrete_jax_profile_grid_uses_host_equivalent_path():
    if not has_jax():
        pytest.skip("JAX is required for concrete JAX array profile coverage")

    indata = InData(
        scalars={
            "PMASS_TYPE": "two_power",
            "PIOTA_TYPE": "power_series",
            "PCURR_TYPE": "cubic_spline_ip",
            "AM": [4.0, 2.0, 1.0],
            "AI": [0.25, 0.5],
            "AC": [1.0],
            "AC_AUX_S": [0.0, 0.5, 1.0],
            "AC_AUX_F": [0.0, 1.0, 0.0],
            "PRES_SCALE": 2.0,
            "BLOAT": 1.0,
            "SPRES_PED": 1.0,
            "LRFP": False,
            "NCURR": 1,
        },
        indexed={},
    )
    s_np = np.linspace(0.0, 1.0, 7)
    prof_np = eval_profiles(indata, s_np)
    prof_jax = eval_profiles(indata, jnp.asarray(s_np))

    for key in ("pressure_pa", "pressure", "iota", "current"):
        np.testing.assert_allclose(np.asarray(prof_jax[key]), np.asarray(prof_np[key]), rtol=1.0e-12, atol=1.0e-12)


def test_profile_coeff_padding_stays_jit_differentiable_for_traced_current_coeffs():
    if not has_jax():
        pytest.skip("JAX is required for traced profile coefficient coverage")

    @jax.jit
    def enclosed_current(coeffs, x):
        padded = profiles_mod._coeff_array(coeffs, nmin=4)
        return profiles_mod._pcurr_power_series_ip(padded, x)

    coeffs = jnp.asarray([1.0, 2.0])
    x = jnp.asarray([0.0, 0.5, 1.0])
    current = enclosed_current(coeffs, x)
    grad = jax.grad(lambda c: jnp.sum(enclosed_current(c, x)))(coeffs)

    np.testing.assert_allclose(np.asarray(current), [0.0, 0.75, 2.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(grad), [1.5, 0.625], rtol=1.0e-12, atol=1.0e-12)


def test_traced_profile_paths_cover_two_power_pedestal_lrfp_and_spline_currents():
    if not has_jax():
        pytest.skip("JAX is required for traced profile coverage")

    @jax.jit
    def two_power_profile(am, ai, ac, s):
        cfg = ProfileInputs(
            pmass_type="two_power",
            piota_type="power_series",
            pcurr_type="two_power",
            am=am,
            ai=ai,
            ac=ac,
            ac_aux_s=jnp.asarray([]),
            ac_aux_f=jnp.asarray([]),
            pres_scale=2.0,
            bloat=1.0,
            spres_ped=0.5,
            lrfp=True,
            ncurr=1,
        )
        prof = eval_profiles(cfg, s)
        return prof["pressure_pa"], prof["iota"], prof["current"]

    @jax.jit
    def spline_profiles(s):
        base = dict(
            pmass_type="power_series",
            piota_type="power_series",
            am=jnp.asarray([0.0]),
            ai=jnp.asarray([]),
            ac=jnp.asarray([1.0]),
            ac_aux_s=jnp.asarray([0.0, 1.0]),
            ac_aux_f=jnp.asarray([0.0, 2.0]),
            pres_scale=1.0,
            bloat=1.0,
            spres_ped=1.0,
            lrfp=False,
            ncurr=1,
        )
        current_ip = eval_profiles(ProfileInputs(pcurr_type="cubic_spline_ip", **base), s)["current"]
        current_i = eval_profiles(ProfileInputs(pcurr_type="cubic_spline_i", **base), s)["current"]
        empty_ip = eval_profiles(
            ProfileInputs(
                pcurr_type="cubic_spline_ip",
                **{**base, "ac_aux_s": jnp.asarray([]), "ac_aux_f": jnp.asarray([])},
            ),
            s,
        )["current"]
        empty_i = eval_profiles(
            ProfileInputs(
                pcurr_type="cubic_spline_i",
                **{**base, "ac_aux_s": jnp.asarray([]), "ac_aux_f": jnp.asarray([])},
            ),
            s,
        )["current"]
        return current_ip, current_i, empty_ip, empty_i

    s = jnp.asarray([0.0, 0.25, 0.75])
    pressure_pa, iota, current = two_power_profile(
        jnp.asarray([10.0, 2.0, 1.0]),
        jnp.asarray([0.0, 2.0]),
        jnp.asarray([3.0, 2.0, 1.0]),
        s,
    )
    expected_pressure = 20.0 * (1.0 - np.asarray(s) ** 2)
    expected_pressure = np.where(np.asarray(s) > 0.5, 20.0 * (1.0 - 0.5**2), expected_pressure)
    np.testing.assert_allclose(np.asarray(pressure_pa), expected_pressure, rtol=1.0e-12, atol=1.0e-12)
    assert np.isinf(np.asarray(iota)[0])
    np.testing.assert_allclose(np.asarray(iota)[1:], 1.0 / (2.0 * np.asarray(s)[1:]), rtol=1.0e-12)
    assert np.all(np.asarray(current) >= 0.0)

    current_ip, current_i, empty_ip, empty_i = spline_profiles(s)
    np.testing.assert_allclose(np.asarray(current_ip), np.asarray(s) ** 2, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(current_i), 2.0 * np.asarray(s), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(empty_ip), np.zeros_like(np.asarray(s)), atol=0.0)
    np.testing.assert_allclose(np.asarray(empty_i), np.zeros_like(np.asarray(s)), atol=0.0)


def test_eager_jax_profile_fallback_matches_vmec_profile_formulas(monkeypatch):
    """Force the differentiable profile path and validate VMEC profile formulas."""
    if not has_jax():
        pytest.skip("JAX is required for eager JAX profile coverage")

    monkeypatch.setattr(profiles_mod, "_can_use_numpy_profile_eval", lambda _s_grid: False)
    s = jnp.asarray([0.0, 0.25, 0.75])
    cfg = ProfileInputs(
        pmass_type="two_power",
        piota_type="power_series",
        pcurr_type="two_power",
        am=jnp.asarray([10.0, 2.0, 1.0]),
        ai=jnp.asarray([0.0, 2.0]),
        ac=jnp.asarray([3.0, 2.0, 1.0]),
        ac_aux_s=jnp.asarray([]),
        ac_aux_f=jnp.asarray([]),
        pres_scale=2.0,
        bloat=1.0,
        spres_ped=0.5,
        lrfp=True,
        ncurr=1,
    )
    prof = eval_profiles(cfg, s)

    expected_pressure = 20.0 * (1.0 - np.asarray(s) ** 2)
    expected_pressure = np.where(np.asarray(s) > 0.5, 20.0 * (1.0 - 0.5**2), expected_pressure)
    np.testing.assert_allclose(np.asarray(prof["pressure_pa"]), expected_pressure, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(prof["pressure"]), profiles_mod.MU0 * expected_pressure, rtol=1.0e-12)
    assert np.isinf(np.asarray(prof["iota"])[0])
    np.testing.assert_allclose(np.asarray(prof["iota"])[1:], 1.0 / (2.0 * np.asarray(s)[1:]), rtol=1.0e-12)
    assert np.all(np.asarray(prof["current"]) >= 0.0)

    spline_base = dict(
        pmass_type="power_series",
        piota_type="power_series",
        am=jnp.asarray([0.0]),
        ai=jnp.asarray([]),
        ac=jnp.asarray([1.0]),
        ac_aux_s=jnp.asarray([0.0, 1.0]),
        ac_aux_f=jnp.asarray([0.0, 2.0]),
        pres_scale=1.0,
        bloat=1.0,
        spres_ped=1.0,
        lrfp=False,
        ncurr=1,
    )
    current_ip = eval_profiles(ProfileInputs(pcurr_type="cubic_spline_ip", **spline_base), s)["current"]
    current_i = eval_profiles(ProfileInputs(pcurr_type="cubic_spline_i", **spline_base), s)["current"]
    np.testing.assert_allclose(np.asarray(current_ip), np.asarray(s) ** 2, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(current_i), 2.0 * np.asarray(s), rtol=1.0e-12, atol=1.0e-12)

    empty_base = {**spline_base, "ac_aux_s": jnp.asarray([]), "ac_aux_f": jnp.asarray([])}
    empty_ip = eval_profiles(ProfileInputs(pcurr_type="cubic_spline_ip", **empty_base), s)["current"]
    empty_i = eval_profiles(ProfileInputs(pcurr_type="cubic_spline_i", **empty_base), s)["current"]
    np.testing.assert_allclose(np.asarray(empty_ip), np.zeros_like(np.asarray(s)), atol=0.0)
    np.testing.assert_allclose(np.asarray(empty_i), np.zeros_like(np.asarray(s)), atol=0.0)


def test_eager_jax_profile_path_reports_unsupported_types_and_power_series_current(monkeypatch):
    if not has_jax():
        pytest.skip("JAX is required for eager JAX profile coverage")

    monkeypatch.setattr(profiles_mod, "_can_use_numpy_profile_eval", lambda _s_grid: False)
    s = jnp.asarray([0.0, 0.5, 1.0])

    current = eval_profiles(
        ProfileInputs(
            pmass_type="power_series",
            piota_type="power_series",
            pcurr_type="power_series",
            am=jnp.asarray([0.0]),
            ai=jnp.asarray([]),
            ac=jnp.asarray([2.0, 3.0]),
            ac_aux_s=jnp.asarray([]),
            ac_aux_f=jnp.asarray([]),
            pres_scale=1.0,
            bloat=1.0,
            spres_ped=1.0,
            lrfp=False,
            ncurr=1,
        ),
        s,
    )["current"]
    np.testing.assert_allclose(np.asarray(current), 2.0 * np.asarray(s) + 1.5 * np.asarray(s) ** 2)

    base = dict(
        pmass_type="power_series",
        piota_type="power_series",
        pcurr_type="power_series",
        am=jnp.asarray([0.0]),
        ai=jnp.asarray([]),
        ac=jnp.asarray([]),
        ac_aux_s=jnp.asarray([]),
        ac_aux_f=jnp.asarray([]),
        pres_scale=1.0,
        bloat=1.0,
        spres_ped=1.0,
        lrfp=False,
        ncurr=1,
    )
    with pytest.raises(NotImplementedError, match="pmass_type"):
        eval_profiles(ProfileInputs(**{**base, "pmass_type": "unsupported"}), s)
    with pytest.raises(NotImplementedError, match="piota_type"):
        eval_profiles(ProfileInputs(**{**base, "piota_type": "unsupported", "ai": jnp.asarray([0.1])}), s)
    with pytest.raises(NotImplementedError, match="pcurr_type"):
        eval_profiles(ProfileInputs(**{**base, "pcurr_type": "unsupported", "ac": jnp.asarray([1.0])}), s)
