from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _booz_like(*, xm, xn, coeffs, iota=0.4, nfp=1):
    return {
        "bmnc_b": np.asarray([coeffs], dtype=float),
        "ixm_b": np.asarray(xm, dtype=float),
        "ixn_b": np.asarray(xn, dtype=float),
        "iota_b": np.asarray([iota], dtype=float),
        "nfp_b": np.asarray(nfp),
    }


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def test_qi_diagnostic_scalar_helpers_cover_unavailable_and_subset_branches() -> None:
    from vmec_jax.quasi_isodynamic.diagnostics import (
        QIDiagnosticOptions,
        _failure_message,
        _finite_float,
        _first_float,
        _handle_error,
        _legacy_nphi_out,
        _list_or_none,
        _max_float,
        _mean_nonaxis_float,
        _min_float,
        _nfp_from_boozer_output,
        _normalized_excess,
        _surface_subset,
        _surface_subset_weights,
    )

    class BadArray:
        def __array__(self, dtype=None, copy=None):
            raise TypeError("not array-like")

    assert _first_float(None) is None
    assert _first_float([]) is None
    assert _finite_float(BadArray()) is None
    assert _finite_float([]) is None
    assert _finite_float([np.inf]) is None
    assert _max_float(None) is None
    assert _max_float([]) is None
    assert _min_float(None) is None
    assert _min_float([]) is None
    assert _list_or_none(None) is None
    assert _list_or_none([]) == []
    assert _list_or_none(np.asarray([1, 2], dtype=np.int64)) == [1, 2]
    assert _list_or_none(np.asarray([1.25, 2.5])) == [1.25, 2.5]
    assert _mean_nonaxis_float(None) is None
    assert _mean_nonaxis_float([]) is None
    assert _mean_nonaxis_float([np.nan]) is None
    assert _mean_nonaxis_float([0.0, 2.0, 4.0]) == pytest.approx(3.0)
    assert _nfp_from_boozer_output({}, None) is None
    assert _nfp_from_boozer_output({"nfp_b": np.asarray([])}, None) is None
    assert _nfp_from_boozer_output({"nfp_b": np.asarray([3])}, None) == 3
    assert _legacy_nphi_out(QIDiagnosticOptions(nphi=11, legacy_nphi_out=19)) == 19
    assert _legacy_nphi_out(QIDiagnosticOptions(nphi=17, legacy_nphi_out=None)) == 401

    record: dict[str, object] = {}
    _handle_error(record, "smooth_error", RuntimeError("bad"), fail_on_error=False)
    assert record["smooth_error"] == "RuntimeError: bad"
    with pytest.raises(RuntimeError, match="bad"):
        _handle_error({}, "smooth_error", RuntimeError("bad"), fail_on_error=True)

    booz = {
        "bmnc_b": np.arange(6.0).reshape(3, 2),
        "iota_b": np.asarray([0.1, 0.2, 0.3]),
        "scalar": np.asarray(7.0),
    }
    assert _surface_subset(booz, None) is booz
    subset = _surface_subset(booz, -1)
    np.testing.assert_allclose(subset["bmnc_b"], [[4.0, 5.0]])
    np.testing.assert_allclose(subset["iota_b"], [0.3])
    np.testing.assert_allclose(subset["scalar"], 7.0)
    with pytest.raises(ValueError, match="surface dimension"):
        _surface_subset({"bmnc_b": np.asarray(1.0)}, 0)
    with pytest.raises(ValueError, match="outside"):
        _surface_subset(booz, 9)

    assert _surface_subset_weights(None, booz=booz, surface_index=-1) is None
    assert _surface_subset_weights([1.0, 2.0], booz=booz, surface_index=-1) == [1.0, 2.0]
    assert _surface_subset_weights([1.0, 2.0, 3.0], booz=booz, surface_index=-1) == [3.0]
    assert _normalized_excess(None, 1.0) is None
    assert _normalized_excess(2.0, None) == 2.0
    assert _normalized_excess(2.0, 4.0) == 0.5
    assert _failure_message("mirror", None, 0.2, upper=True) == "mirror is unavailable"
    assert _failure_message("mirror", 0.3, None, upper=True) == "mirror gate is disabled"
    assert _failure_message("iota", 0.1, 0.4, upper=False) == "iota=0.1 is below target 0.4"


@pytest.mark.py311_slow_coverage
def test_qi_diagnostics_from_boozer_output_records_core_metrics():
    pytest.importorskip("jax")

    from vmec_jax import QI_DIAGNOSTIC_VERSION, QIDiagnosticOptions
    from vmec_jax.quasi_isodynamic.diagnostics import qi_diagnostics_from_boozer_output

    booz = _booz_like(xm=[0, 0], xn=[0, 1], coeffs=[1.0, 0.1])
    options = QIDiagnosticOptions(
        nphi=21,
        nalpha=5,
        n_bounce=5,
        legacy_nphi_out=41,
        mirror_threshold=0.05,
        mirror_ntheta=16,
        mirror_nphi=32,
        phimin=0.2,
    )

    record = qi_diagnostics_from_boozer_output(booz, options=options)

    assert record["qi_diagnostic_version"] == QI_DIAGNOSTIC_VERSION
    assert record["qi_diagnostic_source"] == "boozer"
    assert np.isfinite(record["qi_smooth_total"])
    assert record["qi_raw_total"] == record["qi_smooth_total"]
    assert np.isfinite(record["qi_legacy_total"])
    assert record["qi_mirror_ratio_max"] > 0.09
    assert record["qi_mirror_excess_max"] > 0.04
    assert record["qi_max_elongation"] is None
    assert record["qi_lgradb_enabled"] is False
    assert record["qi_phimin"] == 0.2
    assert record["qi_mboz"] == 0
    assert record["qi_nboz"] == 1
    assert record["qi_boozer_resolution"] == {"mboz": 0, "nboz": 1}


def test_boozer_qi_diagnostic_ranking_prefers_qi_and_flags_mirror_regressions():
    pytest.importorskip("jax")

    from vmec_jax.quasi_isodynamic.diagnostics import (
        QIDiagnosticOptions,
        QISeedSuitabilityTargets,
        qi_diagnostics_from_boozer_output,
        rank_qi_seed_records,
    )

    options = QIDiagnosticOptions(
        nphi=21,
        nalpha=5,
        n_bounce=5,
        legacy_nphi_out=41,
        mirror_threshold=0.08,
        mirror_ntheta=12,
        mirror_nphi=12,
        fail_on_error=True,
    )
    targets = QISeedSuitabilityTargets(
        smooth_qi_max=0.11,
        legacy_qi_max=0.10,
        target_aspect=5.0,
        aspect_relative_tolerance=0.05,
        abs_iota_min=0.41,
        mirror_ratio_max=0.08,
        max_elongation=None,
    )

    def record(label: str, *, xm: list[int], xn: list[int], coeffs: list[float]) -> dict[str, object]:
        out = qi_diagnostics_from_boozer_output(
            {
                "bmnc_b": np.asarray([coeffs], dtype=float),
                "ixm_b": np.asarray(xm, dtype=float),
                "ixn_b": np.asarray(xn, dtype=float),
                "iota_b": np.asarray([0.45], dtype=float),
                "nfp_b": np.asarray(1),
            },
            options=options,
        )
        out.update({"label": label, "aspect": 5.0, "mean_iota": 0.45})
        return out

    good = record("n_only_qi", xm=[0, 0], xn=[0, 1], coeffs=[1.0, 0.03])
    mixed = record("mixed_modes", xm=[0, 0, 1], xn=[0, 1, 0], coeffs=[1.0, 0.03, 0.03])
    mirror_bad = record("mirror_regression", xm=[0, 0, 1], xn=[0, 1, 0], coeffs=[1.0, 0.03, 0.06])
    m_mode_bad = record("m_mode_bad_qi", xm=[0, 1], xn=[0, 0], coeffs=[1.0, 0.03])

    assert good["qi_smooth_total"] < mixed["qi_smooth_total"] < m_mode_bad["qi_smooth_total"]
    assert good["qi_legacy_total"] < mixed["qi_legacy_total"] < m_mode_bad["qi_legacy_total"]
    assert good["qi_mirror_ratio_max"] < mixed["qi_mirror_ratio_max"] < mirror_bad["qi_mirror_ratio_max"]

    ranked = rank_qi_seed_records([mirror_bad, m_mode_bad, mixed, good], targets=targets)
    assert [row["label"] for row in ranked] == ["n_only_qi", "mixed_modes", "mirror_regression", "m_mode_bad_qi"]
    assert ranked[0]["qi_suitability_rank"] == 1
    assert ranked[0]["qi_mirror_gate_passed"] is True
    assert ranked[0]["qi_engineering_gate_passed"] is True
    assert ranked[0]["qi_smooth_rank"] == 1
    assert ranked[0]["qi_legacy_rank"] == 1
    assert ranked[0]["qi_mirror_rank"] == 1
    assert ranked[0]["qi_iota_rank"] == 1

    assert ranked[2]["label"] == "mirror_regression"
    assert ranked[2]["qi_seed_gate_passed"] is True
    assert ranked[2]["qi_mirror_gate_passed"] is False
    assert ranked[2]["qi_engineering_gate_passed"] is False
    assert "mirror" in ranked[2]["qi_gate_failures"]
    assert ranked[2]["qi_mirror_excess_max"] == pytest.approx(0.01)


def test_qi_diagnostics_records_legacy_failure_without_losing_smooth_metric(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.quasi_isodynamic.diagnostics as qid

    def fail_legacy(*_args, **_kwargs):
        raise RuntimeError("legacy unavailable")

    monkeypatch.setattr(qid, "legacy_qi_branch_shuffle_diagnostic_from_boozer_output", fail_legacy)
    booz = _booz_like(xm=[0, 0], xn=[0, 1], coeffs=[1.0, 0.1])

    record = qid.qi_diagnostics_from_boozer_output(
        booz,
        options=qid.QIDiagnosticOptions(nphi=21, nalpha=5, n_bounce=5, legacy_nphi_out=41),
    )

    assert np.isfinite(record["qi_smooth_total"])
    assert record["qi_legacy_total"] is None
    assert "RuntimeError: legacy unavailable" == record["qi_legacy_error"]


def test_qi_diagnostics_fail_on_error_raises_and_records_mirror_subset_error(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.quasi_isodynamic.diagnostics as qid

    booz = _booz_like(xm=[0, 0], xn=[0, 2], coeffs=[1.0, 0.1], nfp=2)

    def fake_smooth(*_args, **_kwargs):
        return {"total": np.asarray(0.0)}

    def fail_mirror(*_args, **_kwargs):
        raise RuntimeError("mirror unavailable")

    monkeypatch.setattr(qid, "quasi_isodynamic_residual_from_boozer_output", fake_smooth)
    monkeypatch.setattr(qid, "mirror_ratio_penalty_from_boozer_output", fail_mirror)

    soft_record = qid.qi_diagnostics_from_boozer_output(
        booz,
        options=qid.QIDiagnosticOptions(
            include_legacy=False,
            mirror_surface_index=3,
            mirror_threshold=0.2,
            fail_on_error=False,
        ),
    )
    assert soft_record["qi_smooth_total"] == 0.0
    assert "ValueError: mirror_surface_index 3 is outside" in soft_record["qi_mirror_error"]
    assert soft_record["qi_nfp"] == 2
    assert soft_record["qi_nboz"] == 1

    with pytest.raises(ValueError, match="mirror_surface_index 3 is outside"):
        qid.qi_diagnostics_from_boozer_output(
            booz,
            options=qid.QIDiagnosticOptions(
                include_legacy=False,
                mirror_surface_index=3,
                fail_on_error=True,
            ),
        )

    with pytest.raises(RuntimeError, match="mirror unavailable"):
        qid.qi_diagnostics_from_boozer_output(
            booz,
            options=qid.QIDiagnosticOptions(include_legacy=False, fail_on_error=True),
        )


def test_qi_diagnostics_from_boozer_output_records_fast_error_and_resolution_edges(monkeypatch):
    import vmec_jax.quasi_isodynamic.diagnostics as qid

    mirror_calls = []

    def fail_smooth(*_args, **_kwargs):
        raise RuntimeError("smooth unavailable")

    def fake_mirror(booz_arg, **kwargs):
        mirror_calls.append((booz_arg, kwargs))
        return {"mirror_ratio": np.asarray([], dtype=float)}

    monkeypatch.setattr(qid, "quasi_isodynamic_residual_from_boozer_output", fail_smooth)
    monkeypatch.setattr(qid, "mirror_ratio_penalty_from_boozer_output", fake_mirror)

    unnormalized = qid.qi_diagnostics_from_boozer_output(
        {
            "bmnc_b": np.asarray([[1.0, 0.1]], dtype=float),
            "ixm_b": np.asarray([], dtype=float),
            "ixn_b": np.asarray([0.0, 3.0], dtype=float),
        },
        nfp=2,
        options=qid.QIDiagnosticOptions(include_legacy=False),
    )

    assert unnormalized["qi_nfp"] == 2
    assert unnormalized["qi_mboz"] is None
    assert unnormalized["qi_nboz"] == 3
    assert unnormalized["qi_smooth_total"] is None
    assert unnormalized["qi_smooth_error"] == "RuntimeError: smooth unavailable"
    assert unnormalized["qi_mirror_ratio_by_surface"] == []
    assert unnormalized["qi_mirror_ratio_max"] is None
    assert unnormalized["qi_mirror_excess_max"] is None

    missing_nfp = qid.qi_diagnostics_from_boozer_output(
        {
            "bmnc_b": np.asarray([[1.0, 0.1]], dtype=float),
            "ixm_b": np.asarray([0.0, 2.0], dtype=float),
            "ixn_b": np.asarray([0.0, 4.0], dtype=float),
        },
        options=qid.QIDiagnosticOptions(include_legacy=False),
    )

    assert missing_nfp["qi_nfp"] is None
    assert missing_nfp["qi_mboz"] == 2
    assert missing_nfp["qi_nboz"] == 4


def test_qi_diagnostics_negative_surface_subset_preserves_mismatched_weights(monkeypatch):
    import vmec_jax.quasi_isodynamic.diagnostics as qid

    calls = {}

    def fake_smooth(*_args, **_kwargs):
        return {"total": np.asarray(0.0)}

    def fake_mirror(booz_arg, **kwargs):
        calls["booz"] = booz_arg
        calls["kwargs"] = kwargs
        return {"mirror_ratio": np.asarray([0.19])}

    monkeypatch.setattr(qid, "quasi_isodynamic_residual_from_boozer_output", fake_smooth)
    monkeypatch.setattr(qid, "mirror_ratio_penalty_from_boozer_output", fake_mirror)

    record = qid.qi_diagnostics_from_boozer_output(
        {
            "bmnc_b": np.asarray([[1.0, 0.1], [1.1, 0.2]], dtype=float),
            "ixm_b": np.asarray([0.0, 1.0], dtype=float),
            "ixn_b": np.asarray([0.0, 2.0], dtype=float),
            "nfp_b": np.asarray([2]),
        },
        weights=[7.0],
        options=qid.QIDiagnosticOptions(
            include_legacy=False,
            mirror_surface_index=-1,
            mirror_threshold=0.21,
        ),
    )

    np.testing.assert_allclose(calls["booz"]["bmnc_b"], [[1.1, 0.2]])
    assert calls["kwargs"]["weights"] == [7.0]
    assert record["qi_mirror_surface_index"] == -1
    assert record["qi_mirror_ratio_max"] == pytest.approx(0.19)
    assert record["qi_mirror_excess_max"] == 0.0


def test_qi_diagnostics_from_state_wraps_existing_components_without_solves(monkeypatch):
    import vmec_jax.quasi_isodynamic.diagnostics as qid

    calls = {}
    booz = {
        "bmnc_b": np.asarray([[1.0, 0.1], [1.1, 0.2]], dtype=float),
        "ixm_b": np.asarray([0, 0], dtype=float),
        "ixn_b": np.asarray([0, 2], dtype=float),
        "iota_b": np.asarray([0.4, 0.5], dtype=float),
        "nfp_b": np.asarray(2),
    }

    def fake_smooth(**kwargs):
        calls["smooth"] = kwargs
        return {
            "total": np.asarray(1.25),
            "booz": booz,
            "surfaces": np.asarray(kwargs["surfaces"], dtype=float),
            "surface_indices": np.asarray([3]),
        }

    def fake_mirror(booz_arg, **kwargs):
        calls["mirror"] = (booz_arg, kwargs)
        return {"mirror_ratio": np.asarray([0.32])}

    def fake_legacy(booz_arg, **kwargs):
        calls["legacy"] = (booz_arg, kwargs)
        return {"total": 2.5, "residual_size": 77}

    def fake_elongation(**kwargs):
        calls["elongation"] = kwargs
        return {"max_elongation": np.asarray(9.5)}

    def fake_lgradb(**kwargs):
        calls["lgradb"] = kwargs
        return {
            "L_grad_B": np.asarray([[0.22, 0.35]]),
            "excess": np.asarray([[1.0, -0.2]]),
        }

    def fake_aspect(**kwargs):
        calls["aspect"] = kwargs
        return np.asarray(5.2)

    def fake_iota_profiles(**kwargs):
        calls["iota"] = kwargs
        return None, np.asarray([0.0, -0.4, -0.5]), None

    monkeypatch.setattr(qid, "quasi_isodynamic_residual_from_state", fake_smooth)
    monkeypatch.setattr(qid, "mirror_ratio_penalty_from_boozer_output", fake_mirror)
    monkeypatch.setattr(qid, "legacy_qi_branch_shuffle_diagnostic_from_boozer_output", fake_legacy)
    monkeypatch.setattr(qid, "max_elongation_penalty_from_state", fake_elongation)
    monkeypatch.setattr(qid, "lgradb_penalty_from_state", fake_lgradb)
    monkeypatch.setattr(qid, "equilibrium_aspect_ratio_from_state", fake_aspect)
    monkeypatch.setattr(qid, "equilibrium_iota_profiles_from_state", fake_iota_profiles)

    options = qid.QIDiagnosticOptions(
        surfaces=[0.5],
        mboz=18,
        nboz=19,
        nphi=21,
        nalpha=7,
        n_bounce=5,
        include_bounce_endpoints=True,
        shuffle_profile_nphi_out=43,
        legacy_nphi_out=61,
        mirror_threshold=0.21,
        mirror_surface_index=1,
        elongation_threshold=8.0,
        include_lgradb=True,
        lgradb_threshold=0.30,
    )
    static = SimpleNamespace(cfg=SimpleNamespace(nfp=2))

    record = qid.qi_diagnostics_from_state(
        state="state",
        static=static,
        indata="indata",
        signgs=-1,
        options=options,
        weights=[1.0, 2.0],
        flux_local="flux",
        prof_local="profiles",
        pressure_local="pressure",
        surface_indices=[3],
    )

    assert calls["smooth"]["mboz"] == 18
    assert calls["smooth"]["nboz"] == 19
    assert calls["smooth"]["include_bounce_endpoints"] is True
    assert calls["smooth"]["shuffle_profile_nphi_out"] == 43
    assert calls["smooth"]["surfaces"] == [0.5]
    assert calls["smooth"]["surface_indices"] == [3]
    assert calls["smooth"]["flux_local"] == "flux"
    assert calls["smooth"]["jit_booz"] is True
    assert np.asarray(calls["mirror"][0]["bmnc_b"]).shape == (1, 2)
    np.testing.assert_allclose(calls["mirror"][0]["bmnc_b"], [[1.1, 0.2]])
    assert calls["mirror"][1]["weights"] == [2.0]
    assert calls["legacy"][1]["nphi_out"] == 61
    assert calls["lgradb"]["flux_local"] == "flux"
    assert calls["aspect"]["state"] == "state"
    assert calls["iota"]["signgs"] == -1

    assert record["qi_diagnostic_source"] == "state"
    assert record["qi_smooth_total"] == 1.25
    assert record["qi_legacy_total"] == 2.5
    assert record["qi_shuffle_profile_nphi_out"] == 43
    assert record["qi_mirror_ratio_max"] == 0.32
    assert record["qi_mirror_ratio_by_surface"] == [0.32]
    assert record["qi_mirror_surface_index"] == 1
    assert record["qi_mirror_excess_max"] == pytest.approx(0.11)
    assert record["qi_max_elongation"] == 9.5
    assert record["qi_elongation_excess"] == 1.5
    assert record["aspect"] == 5.2
    assert record["mean_iota"] == pytest.approx(-0.45)
    assert record["qi_lgradb_enabled"] is True
    assert record["qi_lgradb_min"] == 0.22
    assert record["qi_lgradb_excess_max"] == 1.0
    assert record["qi_mboz"] == 18
    assert record["qi_nboz"] == 19
    assert record["qi_include_bounce_endpoints"] is True
    assert record["qi_jit_booz"] is True
    assert record["qi_surfaces"] == [0.5]
    assert record["qi_surface_indices"] == [3]


def test_qi_diagnostics_from_state_requires_surfaces_and_records_component_errors(monkeypatch):
    import vmec_jax.quasi_isodynamic.diagnostics as qid

    static = SimpleNamespace(cfg=SimpleNamespace(nfp=2))

    with pytest.raises(ValueError, match="surfaces must be supplied"):
        qid.qi_diagnostics_from_state(state="state", static=static, indata="indata", signgs=1)

    def fail_smooth(**_kwargs):
        raise RuntimeError("smooth failed")

    def fail_aspect(**_kwargs):
        raise RuntimeError("aspect failed")

    def fail_iota(**_kwargs):
        raise RuntimeError("iota failed")

    def fail_elongation(**_kwargs):
        raise RuntimeError("elongation failed")

    def fail_lgradb(**_kwargs):
        raise RuntimeError("lgradb failed")

    monkeypatch.setattr(qid, "quasi_isodynamic_residual_from_state", fail_smooth)
    monkeypatch.setattr(qid, "equilibrium_aspect_ratio_from_state", fail_aspect)
    monkeypatch.setattr(qid, "equilibrium_iota_profiles_from_state", fail_iota)
    monkeypatch.setattr(qid, "max_elongation_penalty_from_state", fail_elongation)
    monkeypatch.setattr(qid, "lgradb_penalty_from_state", fail_lgradb)

    record = qid.qi_diagnostics_from_state(
        state="state",
        static=static,
        indata="indata",
        signgs=1,
        surfaces=[0.5],
        options=qid.QIDiagnosticOptions(include_lgradb=True, fail_on_error=False),
    )

    assert record["qi_smooth_error"] == "RuntimeError: smooth failed"
    assert record["qi_aspect_error"] == "RuntimeError: aspect failed"
    assert record["qi_iota_error"] == "RuntimeError: iota failed"
    assert record["qi_elongation_error"] == "RuntimeError: elongation failed"
    assert record["qi_lgradb_error"] == "RuntimeError: lgradb failed"
    assert record["qi_diagnostic_source"] == "state"
    assert record["qi_nfp"] == 2

    with pytest.raises(RuntimeError, match="smooth failed"):
        qid.qi_diagnostics_from_state(
            state="state",
            static=static,
            indata="indata",
            signgs=1,
            surfaces=[0.5],
            options=qid.QIDiagnosticOptions(fail_on_error=True),
        )


@pytest.mark.py311_slow_coverage
def test_qi_diagnostics_from_bundled_solved_qi_seed_records_state_metrics():
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    pytest.importorskip("booz_xform_jax")

    from vmec_jax._compat import enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.quasi_isodynamic.diagnostics import QIDiagnosticOptions, qi_diagnostics_from_state
    from vmec_jax.static import build_static
    from vmec_jax.wout import read_wout, state_from_wout

    enable_x64(True)
    data_dir = _data_dir()
    if not (data_dir / "wout_QI_stel_seed_3127.nc").exists():
        pytest.skip("Optional WOUT fixtures are missing. Run tools/fetch_assets.py --bundle wout-fixtures.")

    cfg, indata = load_config(str(data_dir / "input.QI_stel_seed_3127"))
    static = build_static(cfg)
    wout = read_wout(data_dir / "wout_QI_stel_seed_3127.nc")

    options = QIDiagnosticOptions(
        surfaces=(0.5,),
        mboz=4,
        nboz=4,
        nphi=17,
        nalpha=5,
        n_bounce=5,
        include_bounce_endpoints=True,
        legacy_nphi_out=33,
        mirror_threshold=0.21,
        mirror_ntheta=12,
        mirror_nphi=12,
        mirror_surface_index=0,
        elongation_threshold=8.0,
        elongation_ntheta=12,
        elongation_nphi=6,
        fail_on_error=True,
    )

    record = qi_diagnostics_from_state(
        state=state_from_wout(wout),
        static=static,
        indata=indata,
        signgs=wout.signgs,
        options=options,
    )

    assert record["qi_diagnostic_source"] == "state"
    assert record["qi_nfp"] == 3
    assert record["qi_mboz"] == 4
    assert record["qi_nboz"] == 4
    assert record["qi_include_bounce_endpoints"] is True
    assert record["qi_surfaces"] == [0.5]
    assert record["qi_surface_indices"] == [14]
    assert record["qi_boozer_resolution"] == {"mboz": 4, "nboz": 4}
    assert record["qi_raw_total"] == record["qi_smooth_total"]
    assert 0.0 < record["qi_smooth_total"] < 0.05
    assert 0.0 < record["qi_legacy_total"] < 0.05
    assert record["qi_legacy_nphi_out"] == 33
    assert 0.0 < record["qi_mirror_ratio_max"] < record["qi_mirror_ratio_target"]
    assert record["qi_mirror_excess_max"] == 0.0
    assert record["qi_max_elongation"] > record["qi_elongation_target"]
    assert record["qi_elongation_excess"] > 0.0
    assert record["qi_lgradb_enabled"] is False
    assert record["aspect"] > 0.0
    assert abs(record["mean_iota"]) > 0.0


def test_qi_seed_suitability_annotation_reports_gate_failures():
    from vmec_jax.quasi_isodynamic.diagnostics import QISeedSuitabilityTargets, annotate_qi_seed_suitability

    targets = QISeedSuitabilityTargets(
        smooth_qi_max=2.0e-3,
        legacy_qi_max=1.0e-3,
        target_aspect=5.0,
        aspect_relative_tolerance=0.35,
        abs_iota_min=0.41,
        mirror_ratio_max=0.21,
        max_elongation=8.0,
    )
    record = annotate_qi_seed_suitability(
        {
            "label": "bad_seed",
            "qi_smooth_total": 5.0e-3,
            "qi_legacy_total": 2.0e-3,
            "qi_mirror_ratio_max": 0.25,
            "qi_max_elongation": 8.5,
            "aspect": 8.0,
            "mean_iota": 0.12,
        },
        targets=targets,
    )

    assert record["qi_seed_suitability"] == "needs_attention"
    assert record["qi_metric_gate_passed"] is False
    assert record["qi_seed_gate_passed"] is False
    assert record["qi_engineering_gate_passed"] is False
    assert record["qi_rank_score"] == pytest.approx(7.0e-3)
    assert record["qi_mirror_excess_max"] == pytest.approx(0.04)
    assert record["qi_elongation_excess"] == pytest.approx(0.5)
    assert record["iota_shortfall"] == pytest.approx(0.29)
    assert record["aspect_relative_error"] == pytest.approx(0.6)
    assert record["qi_gate_failures"] == [
        "smooth_qi",
        "legacy_qi",
        "aspect",
        "iota",
        "mirror",
        "elongation",
    ]
    assert "mirror ratio=0.25 exceeds target 0.21" in record["qi_failure_reasons"]


def test_qi_seed_suitability_annotation_handles_disabled_and_missing_gates():
    from vmec_jax.quasi_isodynamic.diagnostics import (
        QISeedSuitabilityTargets,
        annotate_qi_seed_suitability,
        qi_promotion_score,
        rank_qi_seed_records,
    )

    disabled = QISeedSuitabilityTargets(
        smooth_qi_max=None,
        legacy_qi_max=None,
        target_aspect=None,
        abs_iota_min=None,
        mirror_ratio_max=None,
        max_elongation=None,
    )
    pass_record = annotate_qi_seed_suitability(
        {
            "label": "diagnostic_only",
            "qi_smooth_total": np.nan,
            "qi_legacy_total": None,
            "qi_smooth_error": "RuntimeError: bad Boozer solve",
        },
        targets=disabled,
    )

    assert pass_record["qi_seed_suitability"] == "needs_attention"
    assert pass_record["qi_diagnostic_errors"] == ["qi_smooth_error"]
    assert pass_record["qi_rank_score"] == np.inf
    assert pass_record["qi_constraint_score"] == 0.0
    assert "qi_smooth_error: RuntimeError: bad Boozer solve" in pass_record["qi_failure_reasons"]

    ranked = rank_qi_seed_records(
        [
            {"label": "missing", "qi_smooth_total": None, "qi_legacy_total": None},
            {"label": "finite_bad_constraint", "qi_smooth_total": 2.0, "qi_legacy_total": 0.0, "mean_iota": 0.0},
            {"label": "finite_good", "qi_smooth_total": 1.0, "qi_legacy_total": 0.0, "mean_iota": 0.5},
        ],
        targets=QISeedSuitabilityTargets(
            smooth_qi_max=None,
            legacy_qi_max=None,
            target_aspect=None,
            abs_iota_min=0.41,
            mirror_ratio_max=None,
            max_elongation=None,
        ),
    )

    assert [row["label"] for row in ranked] == ["finite_good", "finite_bad_constraint", "missing"]
    assert ranked[0]["qi_iota_gate_passed"] is True
    assert ranked[1]["qi_iota_gate_passed"] is False
    assert ranked[2]["qi_rank_score"] == np.inf

    promotion_targets = QISeedSuitabilityTargets(
        smooth_qi_max=None,
        legacy_qi_max=None,
        target_aspect=5.0,
        aspect_relative_tolerance=0.05,
        abs_iota_min=0.41,
        mirror_ratio_max=0.21,
        max_elongation=8.0,
    )
    cleaner_engineering = {
        "label": "cleaner_engineering",
        "success": True,
        "qi_raw_total": 2.0e-2,
        "qi_legacy_total": 2.0e-2,
        "aspect_final": 5.02,
        "iota_final": -0.45,
        "qi_mirror_ratio_max": 0.20,
        "qi_max_elongation": 7.5,
        "objective_final": 2.0e-2,
    }
    lower_qi_bad_mirror = {
        "label": "lower_qi_bad_mirror",
        "success": True,
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 1.0e-3,
        "aspect_final": 5.0,
        "iota_final": -0.50,
        "qi_mirror_ratio_max": 0.50,
        "qi_max_elongation": 7.0,
        "objective_final": 1.0e-3,
    }
    assert qi_promotion_score(cleaner_engineering, targets=promotion_targets) < qi_promotion_score(
        lower_qi_bad_mirror,
        targets=promotion_targets,
    )
    assert qi_promotion_score(
        {**cleaner_engineering, "qi_legacy_source": "raw_fallback"},
        targets=promotion_targets,
        require_legacy_source=True,
    ) > qi_promotion_score(
        {**cleaner_engineering, "qi_legacy_source": "legacy"},
        targets=promotion_targets,
        require_legacy_source=True,
    )


def test_qi_seed_suitability_accepts_report_and_optimizer_aliases():
    from vmec_jax.quasi_isodynamic.diagnostics import QISeedSuitabilityTargets, annotate_qi_seed_suitability, rank_qi_seed_records

    targets = QISeedSuitabilityTargets(
        smooth_qi_max=2.0e-3,
        legacy_qi_max=1.0e-3,
        target_aspect=5.0,
        aspect_relative_tolerance=0.05,
        abs_iota_min=0.41,
        mirror_ratio_max=0.21,
        max_elongation=8.0,
    )
    report_style = annotate_qi_seed_suitability(
        {
            "case": "report_style",
            "qi_smooth_total": None,
            "smooth_total": 1.0e-3,
            "legacy_total": 4.0e-4,
            "qi_mirror_ratio_max": np.nan,
            "mirror_ratio_max": 0.18,
            "mirror_ratio_target": 0.21,
            "max_elongation": 7.6,
            "elongation_target": 8.0,
            "aspect_final": 5.02,
            "iota_final": -0.45,
        },
        targets=targets,
    )

    assert report_style["qi_seed_gate_passed"] is True
    assert report_style["qi_engineering_gate_passed"] is True
    assert report_style["qi_rank_score"] == pytest.approx(1.4e-3)
    assert report_style["mean_iota"] == pytest.approx(-0.45)
    assert report_style["abs_mean_iota"] == pytest.approx(0.45)
    assert report_style["qi_mirror_ratio_max"] == pytest.approx(0.18)
    assert report_style["qi_max_elongation"] == pytest.approx(7.6)

    ranked = rank_qi_seed_records(
        [
            {
                "case": "bad_iota_alias",
                "qi_raw_total": 8.0e-4,
                "legacy_total": 3.0e-4,
                "aspect_final": 5.0,
                "iota_final": 0.05,
                "mirror_ratio_max": 0.18,
                "max_elongation": 7.0,
            },
            {
                "case": "good_alias",
                "qi_raw_total": 9.0e-4,
                "legacy_total": 3.0e-4,
                "aspect_final": 5.0,
                "iota_final": -0.45,
                "mirror_ratio_max": 0.18,
                "max_elongation": 7.0,
            },
        ],
        targets=targets,
    )

    assert ranked[0]["case"] == "bad_iota_alias"
    assert ranked[0]["qi_iota_gate_passed"] is False
    assert ranked[0]["iota_shortfall"] == pytest.approx(0.36)
    assert ranked[1]["qi_iota_gate_passed"] is True
    assert ranked[0]["qi_iota_rank"] == 2
    assert ranked[1]["qi_iota_rank"] == 1
    assert ranked[0]["qi_mirror_rank"] == 1
    assert ranked[1]["qi_mirror_rank"] == 2


def test_qi_cleanup_candidate_promotes_only_seed_gate_safe_mirror_improvements():
    from vmec_jax.quasi_isodynamic.diagnostics import QISeedSuitabilityTargets, qi_cleanup_candidate_promotable

    targets = QISeedSuitabilityTargets(
        smooth_qi_max=2.0e-3,
        legacy_qi_max=1.0e-3,
        target_aspect=5.0,
        aspect_relative_tolerance=0.35,
        abs_iota_min=0.41,
        mirror_ratio_max=0.21,
        max_elongation=8.0,
    )
    reference = {"qi_mirror_ratio_max": 0.24}
    candidate = {
        "label": "guarded_cleanup",
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 5.0e-4,
        "qi_mirror_ratio_max": 0.18,
        "qi_max_elongation": 7.0,
        "aspect": 5.1,
        "mean_iota": -0.45,
    }

    record = qi_cleanup_candidate_promotable(candidate, reference=reference, targets=targets)

    assert record["qi_cleanup_promoted"] is True
    assert record["qi_seed_gate_passed"] is True
    assert record["qi_cleanup_candidate_mirror"] == pytest.approx(0.18)
    assert record["qi_cleanup_reference_mirror"] == pytest.approx(0.24)
    assert record["qi_cleanup_rejection_reasons"] == []


def test_qi_cleanup_candidate_rejects_mirror_gains_that_break_qi_gate():
    from vmec_jax.quasi_isodynamic.diagnostics import QISeedSuitabilityTargets, qi_cleanup_candidate_promotable

    targets = QISeedSuitabilityTargets(
        smooth_qi_max=2.0e-3,
        legacy_qi_max=1.0e-3,
        target_aspect=5.0,
        aspect_relative_tolerance=0.35,
        abs_iota_min=0.41,
        mirror_ratio_max=0.21,
        max_elongation=8.0,
    )
    candidate = {
        "label": "low_mirror_bad_qi",
        "qi_smooth_total": 5.0e-3,
        "qi_legacy_total": 5.0e-4,
        "qi_mirror_ratio_max": 0.18,
        "qi_max_elongation": 7.0,
        "aspect": 5.1,
        "mean_iota": -0.45,
    }

    record = qi_cleanup_candidate_promotable(candidate, reference={"qi_mirror_ratio_max": 0.24}, targets=targets)

    assert record["qi_cleanup_promoted"] is False
    assert record["qi_seed_gate_passed"] is False
    assert "smooth_qi" in record["qi_gate_failures"]
    assert record["qi_cleanup_rejection_reasons"] == ["QI seed gate failed (smooth_qi)"]


def test_qi_cleanup_candidate_rejects_worsening_mirror_stage():
    from vmec_jax.quasi_isodynamic.diagnostics import QISeedSuitabilityTargets, qi_cleanup_candidate_promotable

    targets = QISeedSuitabilityTargets(
        smooth_qi_max=2.0e-3,
        legacy_qi_max=1.0e-3,
        target_aspect=5.0,
        aspect_relative_tolerance=0.35,
        abs_iota_min=0.41,
        mirror_ratio_max=0.30,
        max_elongation=8.0,
    )
    candidate = {
        "label": "worse_mirror",
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 5.0e-4,
        "qi_mirror_ratio_max": 0.25,
        "qi_max_elongation": 7.0,
        "aspect": 5.1,
        "mean_iota": -0.45,
    }

    record = qi_cleanup_candidate_promotable(candidate, reference={"qi_mirror_ratio_max": 0.24}, targets=targets)

    assert record["qi_cleanup_promoted"] is False
    assert record["qi_seed_gate_passed"] is True
    assert record["qi_cleanup_rejection_reasons"] == [
        "mirror ratio did not improve: candidate=0.25, reference=0.24"
    ]


def test_qi_cleanup_candidate_can_require_engineering_gate():
    from vmec_jax.quasi_isodynamic.diagnostics import QISeedSuitabilityTargets, qi_cleanup_candidate_promotable

    targets = QISeedSuitabilityTargets(
        smooth_qi_max=2.0e-3,
        legacy_qi_max=1.0e-3,
        target_aspect=5.0,
        abs_iota_min=0.41,
        mirror_ratio_max=0.30,
        max_elongation=8.2,
    )
    candidate = {
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 5.0e-4,
        "qi_mirror_ratio_max": 0.35,
        "qi_max_elongation": 7.0,
        "aspect": 5.1,
        "mean_iota": -0.45,
    }

    record = qi_cleanup_candidate_promotable(
        candidate,
        targets=targets,
        require_engineering_gate=True,
        require_mirror_improvement=False,
    )

    assert record["qi_seed_gate_passed"] is True
    assert record["qi_cleanup_promoted"] is False
    assert record["qi_cleanup_rejection_reasons"] == ["QI engineering gate failed (mirror)"]


def test_qi_cleanup_candidate_reports_missing_mirror_inputs_without_rejecting_qi_core():
    from vmec_jax.quasi_isodynamic.diagnostics import QISeedSuitabilityTargets, qi_cleanup_candidate_promotable

    targets = QISeedSuitabilityTargets(
        smooth_qi_max=2.0e-3,
        legacy_qi_max=1.0e-3,
        target_aspect=5.0,
        abs_iota_min=0.41,
        mirror_ratio_max=None,
        max_elongation=8.0,
    )
    seed_safe = {
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 5.0e-4,
        "qi_max_elongation": 7.0,
        "aspect": 5.1,
        "mean_iota": -0.45,
    }

    missing_candidate = qi_cleanup_candidate_promotable(
        seed_safe,
        reference={"qi_mirror_ratio_max": 0.24},
        targets=targets,
    )
    missing_reference = qi_cleanup_candidate_promotable(
        {**seed_safe, "qi_mirror_ratio_max": 0.18},
        reference={},
        targets=targets,
    )

    assert missing_candidate["qi_seed_gate_passed"] is True
    assert missing_candidate["qi_cleanup_candidate_mirror"] is None
    assert missing_candidate["qi_cleanup_rejection_reasons"] == ["candidate mirror ratio is unavailable"]
    assert missing_reference["qi_cleanup_reference_mirror"] is None
    assert missing_reference["qi_cleanup_rejection_reasons"] == ["reference mirror ratio is unavailable"]


@pytest.mark.py311_slow_coverage
def test_qi_seed_ranking_tracks_legacy_goodman_order_on_synthetic_modes():
    pytest.importorskip("jax")

    from vmec_jax.quasi_isodynamic.diagnostics import (
        QIDiagnosticOptions,
        QISeedSuitabilityTargets,
        qi_diagnostics_from_boozer_output,
        rank_qi_seed_records,
    )

    options = QIDiagnosticOptions(
        nphi=33,
        nalpha=9,
        n_bounce=7,
        include_bounce_endpoints=True,
        legacy_nphi_out=101,
        mirror_threshold=1.0,
        mirror_ntheta=16,
        mirror_nphi=16,
    )
    targets = QISeedSuitabilityTargets(
        smooth_qi_max=None,
        legacy_qi_max=None,
        target_aspect=None,
        abs_iota_min=None,
        mirror_ratio_max=None,
        max_elongation=None,
    )
    cases = [
        ("qi_like", _booz_like(xm=[0, 0], xn=[0, 1], coeffs=[1.0, 0.1])),
        ("mixed_qi_helical", _booz_like(xm=[0, 0, 1], xn=[0, 1, 1], coeffs=[1.0, 0.1, 0.04])),
        ("qh_like", _booz_like(xm=[0, 1], xn=[0, 1], coeffs=[1.0, 0.1])),
    ]

    records = []
    for label, booz in cases:
        record = qi_diagnostics_from_boozer_output(booz, options=options)
        record["label"] = label
        records.append(record)

    ranked = rank_qi_seed_records(records, targets=targets)

    assert [record["label"] for record in ranked] == ["qi_like", "mixed_qi_helical", "qh_like"]
    assert [record["label"] for record in sorted(ranked, key=lambda row: row["qi_legacy_total"])] == [
        "qi_like",
        "mixed_qi_helical",
        "qh_like",
    ]
    assert all(record["qi_seed_suitability"] == "pass" for record in ranked)
    assert ranked[0]["qi_suitability_rank"] == 1
