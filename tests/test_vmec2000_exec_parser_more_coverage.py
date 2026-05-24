from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.vmec2000_exec import (
    _find_threed1_file,
    _infer_case_name,
    _parse_vmec2000_threed1,
    _patch_indata,
    flatten_threed1,
    threed1_fsq_total,
)
from tools.diagnostics.compare_freeb_coils_mgrid_vmec2000 import (
    _vmec2000_underconverged_details,
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
                " ITER FSQR FSQZ FSQL fsqr1 fsqz1 fsql1 DELT0R R00 W",
                " bad row is ignored",
                "  1 1.0D-1 2.0D-1 3.0D-1 4.0D-1 5.0D-1 6.0D-1 7.0D-1 8.0D-1 9.0D-1",
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
    assert second.r00 is None
    assert second.w is None
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
    assert details["last_it"] == 5000
    assert details["niter"] == 5000
    assert details["ftolv"] == pytest.approx(1.0e-8)
    assert details["physical_fsq_total_last"] == pytest.approx(0.004002)
    assert details["preconditioned_fsq_total_last"] == pytest.approx(2.996e-5)
    assert details["preconditioned_fsq_total_over_ftolv"] == pytest.approx(2996.0)
