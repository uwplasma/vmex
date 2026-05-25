from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.vmec2000_exec import (
    _find_threed1_file,
    _infer_case_name,
    _parse_vmec2000_threed1,
    _patch_indata,
    _relative_mgrid_file,
    flatten_threed1,
    threed1_fsq_total,
)
from tools.diagnostics.compare_freeb_coils_mgrid_vmec2000 import (
    _base_payload,
    _classify_vmec2000_result_summary,
    _diagnostic_schedule,
    _free_boundary_summary_from_run,
    _make_freeb_indata,
    _parser,
    _vmec2000_nonzero_status,
    _vmec2000_probe_updates,
    _vmec2000_underconverged_details,
    _vmec2000_summary,
    _wout_file_diagnostics,
)


def test_parse_vmec2000_threed1_ignores_noise_and_flushes_stage_at_eof(tmp_path: Path) -> None:
    threed1 = tmp_path / "threed1.synthetic"
    threed1.write_text(
        "\n".join(
            [
                "preamble that should be ignored",
                "  1 9.9E+99 9.9E+99 9.9E+99 9.9E+99 9.9E+99 9.9E+99 9.9E+99",
                "  NS =  5 NO. FOURIER MODES =  3 FTOLV =  2.5D-09 NITER =  11",
                "header before table is ignored",
                " ITER FSQR FSQZ FSQL fsqr1 fsqz1 fsql1 DELT0R R00 W BETA <M> DEL-BSQ FEDGE",
                " bad row is ignored",
                "  1 1.0D-1 2.0D-1 3.0D-1 4.0D-1 5.0D-1 6.0D-1 7.0D-1 8.0D-1 9.0D-1 1.1D+0 1.2D+0 1.3D+0 1.4D+0",
                "  2 1.0E-2 2.0E-2 3.0E-2 4.0E-2 5.0E-2 6.0E-2 7.0E-2",
                " MHD Energy terminates this table",
                "  3 9.9E+1 9.9E+1 9.9E+1 9.9E+1 9.9E+1 9.9E+1 9.9E+1",
                "  NS =  7 NO. FOURIER MODES =  4 FTOLV =  1.0E-10 NITER =  13",
                " ITER FSQR FSQZ FSQL fsqr1 fsqz1 fsql1 DELT0R",
                "  4 4.0E-3 5.0E-3 6.0E-3 7.0E-3 8.0E-3 9.0E-3 1.0E-1",
            ]
        )
    )

    stages = _parse_vmec2000_threed1(threed1)

    assert [(stage.ns, stage.niter, stage.ftolv) for stage in stages] == [
        (5, 11, pytest.approx(2.5e-9)),
        (7, 13, pytest.approx(1.0e-10)),
    ]
    assert [row.it for row in flatten_threed1(stages)] == [1, 2, 4]
    first, second, third = flatten_threed1(stages)
    assert first.r00 == pytest.approx(0.8)
    assert first.w == pytest.approx(0.9)
    assert first.beta == pytest.approx(1.1)
    assert first.avg_m == pytest.approx(1.2)
    assert first.delbsq == pytest.approx(1.3)
    assert first.fedge == pytest.approx(1.4)
    assert second.r00 is None
    assert second.w is None
    assert second.delbsq is None
    assert third.delt0r == pytest.approx(0.1)
    np.testing.assert_allclose(threed1_fsq_total(flatten_threed1(stages)), [0.6, 0.06, 0.015])


def test_parse_vmec2000_threed1_returns_empty_for_missing_stages(tmp_path: Path) -> None:
    threed1 = tmp_path / "threed1.empty"
    threed1.write_text(
        "\n".join(
            [
                " ITER FSQR FSQZ FSQL fsqr1 fsqz1 fsql1 DELT0R",
                "  1 1.0E-1 2.0E-1 3.0E-1 4.0E-1 5.0E-1 6.0E-1 7.0E-1",
            ]
        )
    )

    assert _parse_vmec2000_threed1(threed1) == []
    assert flatten_threed1([]) == []
    assert threed1_fsq_total([]).dtype == float
    assert threed1_fsq_total([]).shape == (0,)


def test_patch_indata_replaces_inserts_before_terminator_and_preserves_newline() -> None:
    text = "\n".join(
        [
            "title",
            "&INDATA",
            "    niter = 99",
            "    ftol_array = 1.0e-8",
            "/",
            "tail",
            "",
        ]
    )

    patched = _patch_indata(text, updates={"NITER": "7", "NS_ARRAY": "5, 9", "MPOL": "4"})

    assert "    NITER = 7" in patched
    assert "    ftol_array = 1.0e-8" in patched
    assert patched.index("NS_ARRAY = 5, 9") < patched.index("/")
    assert patched.index("MPOL = 4") < patched.index("/")
    assert patched.endswith("\n")
    assert patched.count("NITER = 7") == 1


def test_patch_indata_drops_multiline_array_continuations() -> None:
    text = "\n".join(
        [
            "&INDATA",
            "  NS_ARRAY = 16, 49,",
            "     99,",
            "  NITER_ARRAY = 20,",
            "     40,",
            "  FTOL_ARRAY = 1.0e-8,",
            "     1.0e-10,",
            "  MPOL = 8",
            "/",
            "",
        ]
    )

    patched = _patch_indata(
        text,
        updates={
            "NS_ARRAY": "16",
            "NITER_ARRAY": "8",
            "FTOL_ARRAY": "1.0000000000000000e-08",
        },
    )

    lines = patched.splitlines()
    assert "  NS_ARRAY = 16" in lines
    assert "  NITER_ARRAY = 8" in lines
    assert "  FTOL_ARRAY = 1.0000000000000000e-08" in lines
    assert not any(line.strip() in {"99,", "40,", "1.0e-10,"} for line in lines)
    assert "  MPOL = 8" in lines


def test_patch_indata_handles_unterminated_or_missing_indata_block() -> None:
    unterminated = "&INDATA\n  NITER = 10"
    patched = _patch_indata(unterminated, updates={"NITER": "3", "NTOR": "2"})

    assert patched.splitlines() == ["&INDATA", "  NITER = 3", "  NTOR = 2"]
    assert _patch_indata("not a namelist", updates={"NITER": "3"}) == "not a namelist"


def test_relative_mgrid_file_parser_only_returns_copyable_relative_assets() -> None:
    assert _relative_mgrid_file("&INDATA\n MGRID_FILE = 'mgrid_test.nc'\n/\n") == "mgrid_test.nc"
    assert _relative_mgrid_file('&INDATA\n MGRID_FILE = "subdir/mgrid_test.nc"\n/\n') == "subdir/mgrid_test.nc"
    assert _relative_mgrid_file("&INDATA\n MGRID_FILE = 'DIRECT_COILS'\n/\n") is None
    assert _relative_mgrid_file("&INDATA\n MGRID_FILE = '/tmp/mgrid_test.nc'\n/\n") is None
    assert _relative_mgrid_file("&INDATA\n NITER = 1\n/\n") is None


def test_parse_vmec2000_threed1_skips_signed_negative_reference_header(tmp_path: Path) -> None:
    threed1 = tmp_path / "threed1.signed"
    threed1.write_text(
        "\n".join(
            [
                "  NS =  13 NO. FOURIER MODES =  10 FTOLV =  1.0E-08 NITER =     2",
                " ITER FSQR FSQZ FSQL fsqr fsqz fsql DELT RAX WMHD",
                "  1 1.0E-1 2.0E-1 3.0E-1 4.0E-1 5.0E-1 6.0E-1 7.0E-1 8.0E-1 9.0E-1",
                "  NS =  16 NO. FOURIER MODES =  85 FTOLV =  1.0E-10 NITER =    -1",
                " ITER FSQR FSQZ FSQL fsqr fsqz fsql DELT RAX WMHD",
                "  1 1.0E-2 2.0E-2 3.0E-2 4.0E-2 5.0E-2 6.0E-2 7.0E-2 8.0E-2 9.0E-2",
            ]
        )
    )

    stages = _parse_vmec2000_threed1(threed1)

    assert len(stages) == 1
    assert stages[0].ns == 13
    assert stages[0].niter == 2
    assert [row.it for row in stages[0].rows] == [1]


def test_case_name_and_threed1_discovery_precedence(tmp_path: Path) -> None:
    assert _infer_case_name(Path("/tmp/input.nfp4_QH")) == "nfp4_QH"
    assert _infer_case_name(Path("/tmp/custom.deck")) == "custom.deck"

    assert _find_threed1_file(tmp_path, case="missing") is None

    fallback_b = tmp_path / "threed1_zeta"
    fallback_a = tmp_path / "threed1_alpha"
    fallback_b.write_text("")
    fallback_a.write_text("")
    assert _find_threed1_file(tmp_path, case="missing") == fallback_a

    alt = tmp_path / "threed1_case"
    alt.write_text("")
    assert _find_threed1_file(tmp_path, case="case") == alt

    direct = tmp_path / "threed1.case"
    direct.write_text("")
    assert _find_threed1_file(tmp_path, case="case") == direct


def test_freeb_generated_mgrid_no_wout_summary_marks_underconverged() -> None:
    details = _vmec2000_underconverged_details(
        {
            "fsq_total_last": 0.004002,
            "last_row": {
                "it": 5000,
                "fsqr": 0.00162,
                "fsqz": 0.00140,
                "fsql": 0.000982,
                "fsqr1": 1.29e-6,
                "fsqz1": 1.17e-6,
                "fsql1": 2.75e-5,
                "delt0r": 0.0129,
                "w": 9875.4,
            },
            "stages": [{"ns": 12, "niter": 5000, "ftolv": 1.0e-8, "row_count": 26}],
            "stdout_tail": [" Try increasing NITER"],
            "threed1_tail": [],
        }
    )

    assert details["classification"] == "reached_niter_without_wout"
    assert details["printed_try_increasing_niter"] is True
    assert details["reached_niter"] is True
    assert details["more_iter_returncode"] is False
    assert details["last_it"] == 5000
    assert details["niter"] == 5000
    assert details["ftolv"] == pytest.approx(1.0e-8)
    assert details["physical_fsq_total_last"] == pytest.approx(0.004002)
    assert details["preconditioned_fsq_total_last"] == pytest.approx(2.996e-5)
    assert details["preconditioned_fsq_total_over_ftolv"] == pytest.approx(2996.0)


def test_freeb_generated_mgrid_more_iter_returncode_is_vmec_status_not_crash() -> None:
    summary = {
        "returncode": 2,
        "fsq_total_last": 0.00202,
        "last_row": {
            "it": 54,
            "fsqr": 6.90e-4,
            "fsqz": 6.25e-4,
            "fsql": 7.05e-4,
            "fsqr1": 1.56e-6,
            "fsqz1": 1.89e-6,
            "fsql1": 9.95e-5,
            "delt0r": 0.687,
            "w": 10071.0,
            "delbsq": 1.0,
            "fedge": 0.0,
        },
        "stages": [{"ns": 5, "niter": 300, "ftolv": 1.0e-3, "row_count": 2}],
        "stdout_tail": [" Try increasing NITER"],
        "threed1_tail": [" PARVMEC aborting..."],
    }

    details = _vmec2000_underconverged_details(summary)
    status, reason, help_text = _vmec2000_nonzero_status(summary)

    assert details["classification"] == "vmec2000_more_iter_exit"
    assert details["returncode"] == 2
    assert details["nonzero_returncode"] is True
    assert details["more_iter_returncode"] is True
    assert details["printed_try_increasing_niter"] is True
    assert details["delbsq_last"] == pytest.approx(1.0)
    assert details["delbsq_over_ftolv"] == pytest.approx(1000.0)
    assert details["fedge_last"] == pytest.approx(0.0)
    assert status == "more_iter_exit"
    assert reason == "vmec2000_more_iterations_required"
    assert "more_iter_flag=2" in help_text


def test_freeb_generated_mgrid_more_iter_returncode_with_trace_but_no_print_is_vmec_status() -> None:
    summary = {
        "returncode": 2,
        "last_row": {"it": 54, "fsqr1": 1.0e-6, "fsqz1": 2.0e-6, "fsql1": 3.0e-6},
        "stages": [{"ns": 5, "niter": 300, "ftolv": 1.0e-3, "row_count": 2}],
        "stdout_tail": [],
        "threed1_tail": [],
    }

    details = _vmec2000_underconverged_details(summary)
    status, reason, _ = _vmec2000_nonzero_status(summary)

    assert details["classification"] == "vmec2000_more_iter_exit"
    assert details["printed_try_increasing_niter"] is False
    assert details["last_it"] == 54
    assert status == "more_iter_exit"
    assert reason == "vmec2000_more_iterations_required"


def test_freeb_generated_mgrid_more_iter_with_bare_backtrace_is_not_runtime_error() -> None:
    summary = {
        "returncode": 2,
        "last_row": {"it": 2, "fsqr1": 1.0e-6, "fsqz1": 2.0e-6, "fsql1": 3.0e-6},
        "stages": [{"ns": 7, "niter": 2, "ftolv": 1.0e-8, "row_count": 2}],
        "stdout_tail": [" Try increasing NITER"],
        "stderr_tail": [
            "Could not print backtrace: executable file is not an executable",
            "#0  0x107434947",
            "#1  0x10743536f",
        ],
        "threed1_tail": [" Try increasing NITER", " PARVMEC aborting..."],
    }

    details = _vmec2000_underconverged_details(summary)
    status, reason, _ = _vmec2000_nonzero_status(summary)

    assert details["classification"] == "vmec2000_more_iter_exit"
    assert details["runtime_error_detected"] is False
    assert details["runtime_error_tail"] == []
    assert details["backtrace_detected"] is True
    assert "Could not print backtrace" in "\n".join(details["backtrace_tail"])
    assert status == "more_iter_exit"
    assert reason == "vmec2000_more_iterations_required"


def test_freeb_generated_mgrid_returncode_two_with_runtime_error_is_not_more_iter() -> None:
    summary = {
        "returncode": 2,
        "last_row": {"it": 2, "fsqr1": 1.0e-6, "fsqz1": 2.0e-6, "fsql1": 3.0e-6},
        "stages": [{"ns": 7, "niter": 2, "ftolv": 1.0e-8, "row_count": 2}],
        "stdout_tail": [" Try increasing NITER"],
        "stderr_tail": [
            "At line 34 of file fileout.f",
            "Fortran runtime error: Array bound mismatch for dimension 1 of array 'buffer' (0/28)",
            "Error termination. Backtrace:",
        ],
        "threed1_tail": [],
    }

    details = _vmec2000_underconverged_details(summary)
    status, reason, help_text = _vmec2000_nonzero_status(summary)

    assert details["classification"] == "vmec2000_runtime_error"
    assert details["runtime_error_detected"] is True
    assert "Fortran runtime error" in "\n".join(details["runtime_error_tail"])
    assert details["backtrace_detected"] is False
    assert details["more_iter_returncode"] is True
    assert status == "nonzero_exit"
    assert reason == "vmec2000_runtime_error"
    assert "runtime error" in help_text


def test_freeb_generated_mgrid_nonzero_exit_is_not_labeled_underconverged() -> None:
    details = _vmec2000_underconverged_details(
        {
            "returncode": -11,
            "fsq_total_last": 0.00202,
            "last_row": {
                "it": 54,
                "fsqr": 6.90e-4,
                "fsqz": 6.25e-4,
                "fsql": 7.05e-4,
                "fsqr1": 1.56e-6,
                "fsqz1": 1.89e-6,
                "fsql1": 9.95e-5,
                "delt0r": 0.687,
                "w": 10071.0,
            },
            "stages": [{"ns": 5, "niter": 300, "ftolv": 1.0e-3, "row_count": 2}],
            "stdout_tail": [],
            "threed1_tail": [],
        }
    )

    assert details["classification"] == "vmec2000_nonzero_exit"
    assert details["returncode"] == -11
    assert details["nonzero_returncode"] is True
    assert details["more_iter_returncode"] is False
    assert details["reached_niter"] is False
    assert details["printed_try_increasing_niter"] is False


def test_vmec2000_summary_records_whether_mgrid_was_opened(tmp_path: Path) -> None:
    result = SimpleNamespace(
        workdir=tmp_path,
        input_path=tmp_path / "input.case",
        returncode=0,
        stdout="header\n  Opening vacuum field file: mgrid_test.nc\n",
        stderr="",
        runtime_s=0.25,
        threed1_path=None,
        stages=[],
    )

    summary = _vmec2000_summary(result)

    assert summary["returncode"] == 0
    assert summary["opened_mgrid"] is True


def test_free_boundary_summary_from_run_keeps_nestor_channels_compact() -> None:
    run = SimpleNamespace(
        result=SimpleNamespace(
            diagnostics={
                "free_boundary": {
                    "enabled": True,
                    "couple_edge": True,
                    "ivac": 3,
                    "ivacskip": 0,
                    "nvacskip": 4,
                    "nestor_model": "vmec2000_like_dense_integral",
                    "vacuum_stub": False,
                    "final_nestor_recompute_attempted": True,
                    "final_nestor_recompute_failed": False,
                    "final_nestor_sample_time_s": 0.25,
                    "final_nestor_solve_time_s": 0.5,
                    "last_nestor_diagnostics": {
                        "provider_kind": "direct_coils",
                        "mode": "vmec2000_like_dense_integral",
                        "rhs_mode": "bnormal_unit",
                        "sample_points": 16,
                        "bnormal_rms": 1.5,
                        "gsource_rms": 2.5,
                        "bsqvac_rms": 3.5,
                        "bsqvac_mean": 4.5,
                        "large_array": np.arange(5),
                    },
                },
                "freeb_nestor_sample_time_history": np.asarray([0.0, 0.1, 0.2]),
                "freeb_nestor_trial_failed_history": np.asarray([0, 1, 0]),
            }
        )
    )

    summary = _free_boundary_summary_from_run(run)

    assert summary["available"] is True
    assert summary["enabled"] is True
    assert summary["vacuum_stub"] is False
    assert summary["last_nestor_diagnostics"]["provider_kind"] == "direct_coils"
    assert summary["last_nestor_diagnostics"]["bnormal_rms"] == pytest.approx(1.5)
    assert summary["last_nestor_diagnostics"]["bsqvac_mean"] == pytest.approx(4.5)
    assert "large_array" not in summary["last_nestor_diagnostics"]
    assert summary["last_nestor_diagnostics"]["final_nestor_sample_time_s"] == pytest.approx(0.25)
    assert summary["histories"]["freeb_nestor_sample_time_history"]["nonzero_size"] == 2
    assert summary["histories"]["freeb_nestor_sample_time_history"]["sum"] == pytest.approx(0.3)
    assert summary["histories"]["freeb_nestor_trial_failed_history"]["max"] == pytest.approx(1.0)


def test_freeb_diagnostic_schedule_scalar_fallback_and_indata_arrays() -> None:
    args = _parser().parse_args(["--ns", "9", "--niter", "7", "--ftol", "1e-6", "--mgrid-nphi", "4"])

    assert _diagnostic_schedule(args) == ([9], [7], [1.0e-6])

    indata = _make_freeb_indata(SimpleNamespace(scalars={}), mgrid_file="mgrid_test.nc", args=args)

    assert indata.scalars["NS_ARRAY"] == [9]
    assert indata.scalars["NITER_ARRAY"] == [7]
    assert indata.scalars["FTOL_ARRAY"] == [1.0e-6]
    assert indata.scalars["NITER"] == 7
    assert indata.scalars["FTOL"] == pytest.approx(1.0e-6)
    assert indata.scalars["NZETA"] == 4
    assert indata.scalars["NVACSKIP"] == 4


def test_freeb_diagnostic_schedule_accepts_shared_multigrid_arrays() -> None:
    args = _parser().parse_args(
        [
            "--ns-array",
            "5, 9, 13",
            "--niter-array",
            "20, 40, 60",
            "--ftol-array",
            "1e-6, 1e-8, 1e-10",
            "--mgrid-nphi",
            "6",
        ]
    )

    assert _diagnostic_schedule(args) == ([5, 9, 13], [20, 40, 60], [1.0e-6, 1.0e-8, 1.0e-10])

    indata = _make_freeb_indata(SimpleNamespace(scalars={}), mgrid_file="mgrid_test.nc", args=args)

    assert indata.scalars["NS_ARRAY"] == [5, 9, 13]
    assert indata.scalars["NITER_ARRAY"] == [20, 40, 60]
    assert indata.scalars["FTOL_ARRAY"] == [1.0e-6, 1.0e-8, 1.0e-10]
    assert indata.scalars["NITER"] == 60
    assert indata.scalars["FTOL"] == pytest.approx(1.0e-10)


def test_freeb_diagnostic_schedule_rejects_incomplete_or_unequal_arrays() -> None:
    incomplete = _parser().parse_args(["--ns-array", "5,9"])
    with pytest.raises(SystemExit, match="must be provided together"):
        _diagnostic_schedule(incomplete)

    unequal = _parser().parse_args(
        [
            "--ns-array",
            "5,9",
            "--niter-array",
            "20",
            "--ftol-array",
            "1e-6,1e-8",
        ]
    )
    with pytest.raises(SystemExit, match="must have equal lengths"):
        _diagnostic_schedule(unequal)


def test_freeb_diagnostic_payload_reports_resolved_multigrid_schedule(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "--ns-array",
            "5,9",
            "--niter-array",
            "20,40",
            "--ftol-array",
            "1e-6,1e-8",
            "--mgrid-nphi",
            "4",
        ]
    )

    payload = _base_payload(args, out=tmp_path / "summary.json", workdir=tmp_path / "work")

    assert payload["configuration"]["ns_array"] == [5, 9]
    assert payload["configuration"]["niter_array"] == [20, 40]
    assert payload["configuration"]["ftol_array"] == [1.0e-6, 1.0e-8]
    assert payload["configuration"]["uses_multigrid_schedule"] is True
    assert payload["configuration"]["mixed_vmec2000_schedule_non_promotable"] is False
    assert payload["configuration"]["ns"] == 9
    assert payload["configuration"]["niter"] == 40
    assert payload["configuration"]["ftol"] == pytest.approx(1.0e-8)


def test_freeb_diagnostic_payload_marks_vmec2000_niter_override_non_promotable(tmp_path: Path) -> None:
    args = _parser().parse_args(["--vmec2000-niter", "500", "--activate-fsq", "1e99"])

    payload = _base_payload(args, out=tmp_path / "summary.json", workdir=tmp_path / "work")

    assert payload["configuration"]["vmec2000_niter"] == 500
    assert payload["configuration"]["mixed_vmec2000_schedule_non_promotable"] is True
    assert payload["configuration"]["activate_fsq"] == pytest.approx(1.0e99)
    assert payload["configuration"]["active_free_boundary_requested"] is True


def test_vmec2000_promotion_probe_updates_are_bounded_vmec2000_only_patches() -> None:
    args = _parser().parse_args(
        [
            "--ns-array",
            "5,7",
            "--niter-array",
            "20,40",
            "--ftol-array",
            "1e-6,1e-8",
            "--vmec2000-probe-ftols",
            "1e-2,1e-3",
            "--vmec2000-probe-max-main-iterations",
            "2,4",
        ]
    )

    probes = _vmec2000_probe_updates(args)

    labels = [probe["label"] for probe in probes]
    assert labels == [
        "loose_ftol_0.01",
        "loose_ftol_0.001",
        "force_full3d_output",
        "max_main_iterations_2",
        "max_main_iterations_4",
    ]
    assert probes[0]["updates"]["FTOL"] == "1.0000000000000000e-02"
    assert probes[0]["updates"]["FTOL_ARRAY"] == "1.0000000000000000e-02, 1.0000000000000000e-02"
    assert probes[2]["updates"]["LFULL3D1OUT"] == "T"
    assert probes[2]["updates"]["NITER_ARRAY"] == "20, 40"
    assert probes[3]["updates"]["MAX_MAIN_ITERATIONS"] == "2"


def test_classify_vmec2000_summary_completed_no_wout_and_more_iter(tmp_path: Path) -> None:
    completed = {"returncode": 0, "last_row": {"it": 1}, "stages": [{"niter": 2, "ftolv": 1e-6}]}
    wout = tmp_path / "wout_case.nc"
    wout.write_bytes(b"placeholder")

    _classify_vmec2000_result_summary(completed, wout_path=wout)

    assert completed["status"] == "completed"
    assert completed["wout_path"] == wout

    no_wout = {
        "returncode": 0,
        "last_row": {"it": 2, "fsqr1": 1.0, "fsqz1": 2.0, "fsql1": 3.0},
        "stages": [{"niter": 2, "ftolv": 1e-6}],
        "stdout_tail": [" Try increasing NITER"],
        "threed1_tail": [],
    }
    _classify_vmec2000_result_summary(no_wout, wout_path=tmp_path / "missing.nc")
    assert no_wout["status"] == "no_wout"
    assert no_wout["underconverged"]["classification"] == "reached_niter_without_wout"

    more_iter = {
        "returncode": 2,
        "last_row": {"it": 1, "fsqr1": 1.0, "fsqz1": 2.0, "fsql1": 3.0},
        "stages": [{"niter": 2, "ftolv": 1e-6}],
        "stdout_tail": [],
        "threed1_tail": [],
    }
    _classify_vmec2000_result_summary(more_iter, wout_path=tmp_path / "missing_more.nc")
    assert more_iter["status"] == "more_iter_exit"
    assert more_iter["reason"] == "vmec2000_more_iterations_required"


def test_wout_file_diagnostics_records_scalar_only_error_wout(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    wout = tmp_path / "wout_error.nc"
    with netcdf4.Dataset(wout, "w") as ds:
        ds.createDimension("ext_current", 1)
        ier = ds.createVariable("ier_flag", "i4")
        ier.assignValue(7)
        ds.createVariable("extcur", "f8", ("ext_current",))[:] = [1.0]

    diagnostics = _wout_file_diagnostics(wout)

    assert diagnostics["exists"] is True
    assert diagnostics["size_bytes"] > 0
    assert diagnostics["ier_flag"] == 7
    assert diagnostics["has_mode_table"] is False
    assert "xm" not in diagnostics["variables"]
