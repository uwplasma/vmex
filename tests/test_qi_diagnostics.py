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


def test_qi_diagnostics_from_boozer_output_records_core_metrics():
    pytest.importorskip("jax")

    from vmec_jax import QI_DIAGNOSTIC_VERSION, QIDiagnosticOptions
    from vmec_jax.qi_diagnostics import qi_diagnostics_from_boozer_output

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


def test_qi_diagnostics_records_legacy_failure_without_losing_smooth_metric(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.qi_diagnostics as qid

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


def test_qi_diagnostics_from_state_wraps_existing_components_without_solves(monkeypatch):
    import vmec_jax.qi_diagnostics as qid

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

    monkeypatch.setattr(qid, "quasi_isodynamic_residual_from_state", fake_smooth)
    monkeypatch.setattr(qid, "mirror_ratio_penalty_from_boozer_output", fake_mirror)
    monkeypatch.setattr(qid, "legacy_qi_branch_shuffle_diagnostic_from_boozer_output", fake_legacy)
    monkeypatch.setattr(qid, "max_elongation_penalty_from_state", fake_elongation)
    monkeypatch.setattr(qid, "lgradb_penalty_from_state", fake_lgradb)

    options = qid.QIDiagnosticOptions(
        surfaces=[0.5],
        mboz=18,
        nboz=19,
        nphi=21,
        nalpha=7,
        n_bounce=5,
        include_bounce_endpoints=True,
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
    assert calls["smooth"]["surfaces"] == [0.5]
    assert calls["smooth"]["surface_indices"] == [3]
    assert calls["smooth"]["flux_local"] == "flux"
    assert np.asarray(calls["mirror"][0]["bmnc_b"]).shape == (1, 2)
    np.testing.assert_allclose(calls["mirror"][0]["bmnc_b"], [[1.1, 0.2]])
    assert calls["mirror"][1]["weights"] == [2.0]
    assert calls["legacy"][1]["nphi_out"] == 61
    assert calls["lgradb"]["flux_local"] == "flux"

    assert record["qi_diagnostic_source"] == "state"
    assert record["qi_smooth_total"] == 1.25
    assert record["qi_legacy_total"] == 2.5
    assert record["qi_mirror_ratio_max"] == 0.32
    assert record["qi_mirror_ratio_by_surface"] == [0.32]
    assert record["qi_mirror_surface_index"] == 1
    assert record["qi_mirror_excess_max"] == pytest.approx(0.11)
    assert record["qi_max_elongation"] == 9.5
    assert record["qi_elongation_excess"] == 1.5
    assert record["qi_lgradb_enabled"] is True
    assert record["qi_lgradb_min"] == 0.22
    assert record["qi_lgradb_excess_max"] == 1.0
    assert record["qi_mboz"] == 18
    assert record["qi_nboz"] == 19
    assert record["qi_include_bounce_endpoints"] is True
    assert record["qi_surfaces"] == [0.5]
    assert record["qi_surface_indices"] == [3]


def test_qi_diagnostics_from_bundled_solved_qi_seed_records_state_metrics():
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    pytest.importorskip("booz_xform_jax")

    from vmec_jax._compat import enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.qi_diagnostics import QIDiagnosticOptions, qi_diagnostics_from_state
    from vmec_jax.static import build_static
    from vmec_jax.wout import read_wout, state_from_wout

    enable_x64(True)
    data_dir = _data_dir()
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
