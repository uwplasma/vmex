from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = ROOT / "tools" / "diagnostics" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_readme_fsq_trace_parser_handles_vmec_and_vmecpp_tables():
    mod = _load_tool("readme_fsq_trace")
    stdout = """
      ITER    FSQR      FSQZ      FSQL    RAX(v=0)
        1 | 1.0D-01 | 2.0D-02 | 3.0D-03 | 0.0
        2   4.0E-02   5.0E-03   6.0E-04   0.0
     EXECUTION TERMINATED NORMALLY
    """

    it, fsq = mod._parse_vmec_table_trace(stdout)

    np.testing.assert_array_equal(it, np.asarray([1, 2]))
    np.testing.assert_allclose(fsq, np.asarray([0.123, 0.0456]))


def test_runtime_compare_exports_vmec2000_vmec_jax_and_vmecpp_rows(tmp_path):
    mod = _load_tool("readme_runtime_compare")
    summary = {
        "cases": [
            {
                "id": "case_a",
                "lfreeb": False,
                "lasym": False,
                "axisymmetric": True,
            }
        ],
        "results": [
            {"case_id": "case_a", "backend": "vmec2000", "runtime_s": 4.0, "max_rss_bytes": 100},
            {
                "case_id": "case_a",
                "backend": "vmec_jax",
                "runtime_cold_s": 2.0,
                "runtime_warm_s": 1.0,
                "peak_footprint_bytes": 200,
            },
            {"case_id": "case_a", "backend": "vmecpp", "runtime_s": 0.5, "max_rss_bytes": 50},
        ],
    }
    gpu_summary = {
        "cases": summary["cases"],
        "results": [
            {
                "case_id": "case_a",
                "backend": "vmec_jax",
                "runtime_cold_s": 3.0,
                "runtime_warm_s": 0.25,
                "peak_footprint_bytes": 300,
            }
        ],
    }
    rows = mod._collect_records([summary], [gpu_summary])

    assert len(rows) == 1
    assert rows[0]["vmec_runtime_s"] == 4.0
    assert rows[0]["cpu_runtime_s"] == 2.0
    assert rows[0]["cpu_warm_runtime_s"] == 1.0
    assert rows[0]["gpu_runtime_s"] == 3.0
    assert rows[0]["gpu_warm_runtime_s"] == 0.25
    assert rows[0]["vmecpp_runtime_s"] == 0.5

    csv_path = tmp_path / "runtime.csv"
    json_path = tmp_path / "runtime.json"
    table_path = tmp_path / "runtime.md"
    figure_path = tmp_path / "runtime.png"
    mod._write_csv(rows, csv_path)
    mod._write_json(
        rows,
        json_path,
        cpu_summary_paths=[Path("cpu_summary.json")],
        gpu_summary_paths=[Path("gpu_summary.json")],
        figure_path=figure_path,
        table_path=table_path,
    )

    csv_text = csv_path.read_text()
    assert "vmec_jax_cold_speedup_vs_vmec2000" in csv_text
    assert "vmec_jax_gpu_warm_speedup_vs_cpu_warm" in csv_text
    assert "2.0" in csv_text
    payload = json.loads(json_path.read_text())
    record = payload["records"][0]
    assert record["case_id"] == "case_a"
    assert record["vmec_jax_cold_speedup_vs_vmec2000"] == 2.0
    assert record["vmec_jax_warm_speedup_vs_vmec2000"] == 4.0
    assert record["vmec_jax_gpu_warm_speedup_vs_vmec2000"] == 16.0
    assert record["vmec_jax_gpu_warm_speedup_vs_cpu_warm"] == 4.0
    assert record["vmecpp_speedup_vs_vmec2000"] == 8.0


def test_direct_coil_segmented_replay_report_synthetic_policy_helpers(monkeypatch):
    mod = _load_tool("direct_coil_segmented_replay_report")
    traces = [
        {
            "preconditioner_use_lax_tridi": True,
            "preconditioner_use_precomputed_tridi": False,
            "value": np.asarray([1.0, 2.0]),
        },
        {
            "preconditioner_use_lax_tridi": True,
            "preconditioner_use_precomputed_tridi": False,
            "value": np.asarray([3.0, 4.0]),
        },
        {
            "preconditioner_use_lax_tridi": True,
            "preconditioner_use_precomputed_tridi": False,
            "value": np.asarray([5.0, 6.0]),
        },
    ]

    changed = mod._with_synthetic_policy_segments(traces)

    assert changed[0]["preconditioner_use_lax_tridi"] is True
    assert changed[1]["preconditioner_use_lax_tridi"] is False
    assert changed[2]["preconditioner_use_lax_tridi"] is True
    assert traces[1]["preconditioner_use_lax_tridi"] is True
    assert mod._json_ready({"x": np.asarray([1.0]), "bad": float("nan")}) == {"x": [1.0], "bad": None}

    from vmec_jax import free_boundary_adjoint

    calls = []

    def fake_controller_replay(*_args, **kwargs):
        calls.append(kwargs)
        return {
            "objective": np.asarray(1.0),
            "state": np.asarray([0.0]),
            "preconditioner_controls_stacked": False,
            "preconditioner_controls_segment_stacked": (),
            "used_accepted_only_fast_path": bool(kwargs["use_accepted_only_fast_path"]),
            "accepted_only_fast_path_segments": (bool(kwargs["use_accepted_only_fast_path"]),),
        }

    monkeypatch.setattr(
        free_boundary_adjoint,
        "direct_coil_accepted_trace_controller_replay_objective_jax",
        fake_controller_replay,
    )
    replay, timings = mod._timed_replay(
        params=object(),
        initial_state=object(),
        static=object(),
        traces=[],
        signgs=1,
        use_segments=True,
        use_segment_preconditioner_controls=False,
        use_accepted_only_fast_path=False,
        repeats=1,
    )

    assert replay["used_accepted_only_fast_path"] is False
    assert len(timings) == 1
    assert calls[0]["use_preconditioner_policy_segments"] is True
    assert calls[0]["use_accepted_only_fast_path"] is False


def test_freeb_replay_diagnostic_utils_json_ready_and_timed_call():
    mod = _load_tool("freeb_replay_diagnostic_utils")

    payload = mod.json_ready(
        {
            "array": np.asarray([1.0, 2.0]),
            "scalar": np.float64(3.0),
            "nan": float("nan"),
            "nested": (np.asarray([4.0]),),
        }
    )

    assert payload == {"array": [1.0, 2.0], "scalar": 3.0, "nan": None, "nested": [[4.0]]}
    value, first, warm = mod.timed_call(lambda x: x + 1, 2, warm_repeats=1)
    assert value == 3
    assert first >= 0.0
    assert len(warm) == 1
    assert warm[0] >= 0.0


def test_direct_coil_strict_update_replay_report_helpers():
    mod = _load_tool("direct_coil_strict_update_replay_report")
    tree = {"a": np.asarray([[1.0, 2.0], [3.0, 4.0]]), "b": [np.asarray([5.0, 6.0])]}

    first = mod._first_slice(tree)

    np.testing.assert_allclose(np.asarray(first["a"]), np.asarray([1.0, 2.0]))
    assert np.asarray(first["b"][0]).shape == ()
    assert mod._json_ready({"x": np.asarray([1.0]), "bad": float("inf")}) == {"x": [1.0], "bad": None}


def test_direct_coil_boundary_replay_report_selects_active_trace():
    mod = _load_tool("direct_coil_boundary_replay_report")
    traces = [
        {"freeb_bsqvac_half": None, "freeb_nestor_trace": None},
        {"freeb_bsqvac_half": np.asarray([1.0]), "freeb_nestor_trace": {"br_axis": 0.0}},
        {"freeb_bsqvac_half": np.asarray([2.0]), "freeb_nestor_trace": {"br_axis": 1.0}},
    ]

    index, trace = mod._select_active_trace(traces, -1)

    assert index == 2
    assert trace["freeb_nestor_trace"]["br_axis"] == 1.0
    index, trace = mod._select_active_trace(traces, 0)
    assert index == 2
    assert mod._json_ready({"x": np.asarray([1.0]), "bad": float("-inf")}) == {"x": [1.0], "bad": None}


def test_vmecpp_runtime_two_cases_runtime_updates():
    mod = _load_tool("readme_vmecpp_runtime_two_cases")

    assert mod._runtime_updates(ns=None, niter=None, ftol=None, nstep=None) == {}
    assert mod._runtime_updates(ns=17, niter=25, ftol=1e-9, nstep=1) == {
        "NSTEP": "1",
        "NS_ARRAY": "17",
        "NITER_ARRAY": "25",
        "FTOL_ARRAY": "1.000e-09",
    }
