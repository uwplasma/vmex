from __future__ import annotations

from pathlib import Path

import pytest

from tools.diagnostics.parity_sweep_manifest import DEFAULT_MANIFEST, _parse_manifest
from vmec_jax.namelist import read_indata


REPO_ROOT = DEFAULT_MANIFEST.parents[2]


def _resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _nonzero_indexed_coefficients(indata, name: str) -> list[float]:
    return [float(value) for value in indata.indexed.get(name, {}).values() if abs(float(value)) > 0.0]


def _nonzero_scalar_coefficients(indata, name: str) -> list[float]:
    values = indata.get(name, [])
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    return [float(value) for value in values if abs(float(value)) > 0.0]


def test_enabled_local_manifest_cases_match_input_physics_metadata() -> None:
    """Manifest booleans should agree with the local VMEC inputs they launch."""
    _meta, cases = _parse_manifest(DEFAULT_MANIFEST)
    local_cases = [
        case
        for case in cases
        if case.get("enabled", True) and str(case.get("source")) == "vmec_jax/examples"
    ]
    assert local_cases

    for case in local_cases:
        indata = read_indata(_resolve_repo_path(str(case["input"])))
        assert bool(case["lfreeb"]) is bool(indata.get_bool("LFREEB")), case["id"]
        assert bool(case["lasym"]) is bool(indata.get_bool("LASYM")), case["id"]
        assert int(case["nfp"]) == int(indata.get_int("NFP")), case["id"]
        assert bool(case["axisymmetric"]) is (int(indata.get_int("NTOR")) == 0), case["id"]
        if case.get("compare") == "stage_trace":
            assert int(case["ntor"]) == int(indata.get_int("NTOR")), case["id"]
        else:
            assert 0 <= int(case["ntor"]) <= int(indata.get_int("NTOR")), case["id"]

        if bool(case["lasym"]):
            asymmetric_channels = (
                _nonzero_indexed_coefficients(indata, "RBS")
                + _nonzero_indexed_coefficients(indata, "ZBC")
                + _nonzero_scalar_coefficients(indata, "RAXIS_CS")
                + _nonzero_scalar_coefficients(indata, "ZAXIS_CC")
            )
            assert asymmetric_channels, case["id"]


def test_lasym_finite_beta_smoke_manifest_is_bounded_and_asset_backed() -> None:
    """Keep the required fixed-boundary LASYM=true finite-beta parity lane cheap."""
    _meta, cases = _parse_manifest(DEFAULT_MANIFEST)
    case = next((case for case in cases if case.get("id") == "fixed_nonaxis_lasym_true_basic_non_stellsym"), None)
    assert case is not None

    assert case.get("enabled") is True
    assert case.get("tier") == "smoke"
    assert case.get("compare") == "stage_trace"
    assert case.get("source") == "vmec_jax/examples"
    assert bool(case.get("lfreeb")) is False
    assert bool(case.get("lasym")) is True
    assert bool(case.get("axisymmetric")) is False
    assert int(case.get("single_ns", 0)) == 13
    assert int(case.get("max_iter", 0)) <= 10
    assert int(case.get("vmec_nstep", 0)) == 1
    assert float(case.get("rtol", 1.0)) <= 2.0e-3
    assert float(case.get("vmec_timeout", 0.0)) <= 120.0

    input_path = _resolve_repo_path(str(case["input"]))
    reference_wout = REPO_ROOT / "examples/data/single_grid/wout_basic_non_stellsym_pressure_reference.nc"
    assert input_path.exists()
    if not reference_wout.exists():
        pytest.skip("Optional LASYM finite-beta WOUT fixture is missing. Run tools/fetch_assets.py --bundle wout-fixtures.")

    indata = read_indata(input_path)
    assert bool(indata.get_bool("LASYM")) is True
    assert bool(indata.get_bool("LFREEB")) is False
    assert int(indata.get_int("NTOR")) > 0
    assert float(indata.get_float("GAMMA")) > 0.0
    assert any(abs(float(value)) > 0.0 for value in indata.get("AM", []))
    assert _nonzero_indexed_coefficients(indata, "RBS")
    assert _nonzero_indexed_coefficients(indata, "ZBC")


def test_free_boundary_lasym_manifest_requires_local_mgrid_and_iter_gates() -> None:
    """Guard the self-contained LASYM=true free-boundary planning lane."""
    _meta, cases = _parse_manifest(DEFAULT_MANIFEST)
    case = next((case for case in cases if case.get("id") == "freeb_nonaxis_lasym_true_cth_like_local"), None)
    assert case is not None

    assert case.get("enabled") is True
    assert case.get("tier") == "planning"
    assert case.get("compare") == "freeb_scalpot"
    assert case.get("source") == "vmec_jax/examples"
    assert bool(case.get("lfreeb")) is True
    assert bool(case.get("lasym")) is True
    assert bool(case.get("axisymmetric")) is False

    iter_list = [int(value) for value in case.get("iter_list", [])]
    assert iter_list == [80, 100]
    assert int(case.get("max_iter", 0)) >= max(iter_list)
    assert float(case.get("activate_fsq", 0.0)) == 1.0e99
    assert float(case.get("max_runtime_s", 0.0)) <= 45.0
    assert float(case.get("max_total_runtime_s", 0.0)) <= 95.0

    metric_by_iter = case.get("metric_thresholds_rel_scaled_by_iter", {})
    runtime_by_iter = case.get("runtime_thresholds_s_by_iter", {})
    assert set(metric_by_iter) == {str(iter_idx) for iter_idx in iter_list}
    assert set(runtime_by_iter) == {str(iter_idx) for iter_idx in iter_list}
    assert metric_by_iter["80"]["source_sym"] <= 1.0e-2
    assert metric_by_iter["80"]["bvec_nonsing_fouri"] <= 1.0e-2
    assert all(float(rules["max_runtime_s"]) <= 40.0 for rules in runtime_by_iter.values())

    input_path = _resolve_repo_path(str(case["input"]))
    indata = read_indata(input_path)
    mgrid_name = str(indata.get("MGRID_FILE"))
    assert mgrid_name == "mgrid_cth_like_lasym_small.nc"
    assert (input_path.parent / mgrid_name).exists()
    assert bool(indata.get_bool("LFREEB")) is True
    assert bool(indata.get_bool("LASYM")) is True
    assert len(indata.get("EXTCUR", [])) >= 2
    assert _nonzero_indexed_coefficients(indata, "RBS")
    assert _nonzero_indexed_coefficients(indata, "ZBC")


def test_runtime_threshold_reports_missing_iter_as_failure() -> None:
    """Parity summaries must fail closed when a declared VMEC2000 checkpoint is absent."""
    from tools.diagnostics.parity_sweep_manifest import _evaluate_runtime_thresholds

    ok, report = _evaluate_runtime_thresholds(
        {"runtime_thresholds_s_by_iter": {"80": {"max_runtime_s": 40.0}}},
        [{"iter": 79, "runtime_s": 3.0}],
    )

    assert not ok
    assert report["by_iter"]["80"]["observed_runtime_s"] != report["by_iter"]["80"]["observed_runtime_s"]
    assert report["by_iter"]["80"]["max_runtime_s"]["pass"] is False
