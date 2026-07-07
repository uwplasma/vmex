from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.diagnostics.parity.vmec2000_exec_stage_trace_compare import _parse_float_list_arg, _parse_int_list_arg
from vmec_jax.vmec2000_exec import (
    _default_exec_candidates,
    _find_threed1_file,
    _infer_case_name,
    _parse_vmec2000_threed1,
    _patch_indata,
    _relative_mgrid_file,
    flatten_threed1,
    find_vmec2000_exec,
    run_xvmec2000,
    threed1_fsq_total,
)


def test_stage_trace_compare_list_args_accept_spaces_and_commas() -> None:
    assert _parse_int_list_arg("12 31,50") == [12, 31, 50]
    assert _parse_int_list_arg("  ") is None
    assert _parse_float_list_arg("1e-8 1e-10,1e-12") == [1.0e-8, 1.0e-10, 1.0e-12]
    assert _parse_float_list_arg("") is None


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


def test_parse_vmec2000_threed1_splits_packed_beta_and_avg_m_fields(tmp_path: Path) -> None:
    threed1 = tmp_path / "threed1.packed"
    threed1.write_text(
        "\n".join(
            [
                "  NS =  7 NO. FOURIER MODES =  4 FTOLV =  1.0E-10 NITER =  13",
                " ITER FSQR FSQZ FSQL fsqr1 fsqz1 fsql1 DELT0R R00 W BETA <M> DELBSQ FEDGE",
                "  17 1.0E-1 2.0E-1 3.0E-1 4.0E-1 5.0E-1 6.0E-1 7.0E-1 8.0E-1 9.0E-1 1.001-3.53E+00 4.0E-4 5.0E-5",
            ]
        )
    )

    rows = flatten_threed1(_parse_vmec2000_threed1(threed1))

    assert len(rows) == 1
    row = rows[0]
    assert row.beta == pytest.approx(1.001)
    assert row.avg_m == pytest.approx(-3.53)
    assert row.delbsq == pytest.approx(4.0e-4)
    assert row.fedge == pytest.approx(5.0e-5)


def test_parse_vmec2000_threed1_preserves_missing_exponent_underflow_token(tmp_path: Path) -> None:
    threed1 = tmp_path / "threed1.underflow"
    threed1.write_text(
        "\n".join(
            [
                "  NS =  7 NO. FOURIER MODES =  4 FTOLV =  1.0E-10 NITER =  13",
                " ITER FSQR FSQZ FSQL fsqr1 fsqz1 fsql1 DELT0R R00 W BETA <M>",
                "  2 1.0E-1 2.0E-1 3.0E-1 4.0E-1 5.0E-1 6.0E-1 7.0E-1 8.0E-1 1.0564215887228806-316 1.0E+0 2.0E+0",
            ]
        )
    )

    rows = flatten_threed1(_parse_vmec2000_threed1(threed1))

    assert len(rows) == 1
    assert rows[0].w == pytest.approx(1.0564215887228806e-316)
    assert rows[0].beta == pytest.approx(1.0)
    assert rows[0].avg_m == pytest.approx(2.0)


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


def test_patch_indata_stops_continuation_drop_at_comment() -> None:
    text = "\n".join(
        [
            "&INDATA",
            "  NS_ARRAY = 16,",
            "  ! keep this comment",
            "/",
            "",
        ]
    )

    patched = _patch_indata(text, updates={"NS_ARRAY": "5"})

    assert "  NS_ARRAY = 5" in patched.splitlines()
    assert "  ! keep this comment" in patched.splitlines()


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


def test_vmec2000_exec_helpers_cover_path_discovery_and_relative_mgrid(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    exec_path = root / "STELLOPT" / "VMEC2000" / "Release" / "xvmec2000"
    exec_path.parent.mkdir(parents=True)
    exec_path.write_text("#!/bin/sh\n")

    assert _default_exec_candidates(root, include_user_bin=False)[0] == exec_path
    assert Path.home() / "bin" / "xvmec2000" in _default_exec_candidates(root, include_user_bin=True)
    assert find_vmec2000_exec(root=root) == exec_path

    env_exec = tmp_path / "env_xvmec2000"
    env_exec.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VMEC2000_EXEC", str(env_exec))
    assert find_vmec2000_exec(root=root) == env_exec

    assert _relative_mgrid_file("MGRID_FILE = 'mgrid_test.nc'") == "mgrid_test.nc"
    assert _relative_mgrid_file("MGRID_FILE = 'NONE'") is None
    assert _relative_mgrid_file("MGRID_FILE = 'DIRECT_COILS'") is None
    assert _relative_mgrid_file(f"MGRID_FILE = '{tmp_path / 'absolute.nc'}'") is None
    assert _relative_mgrid_file("&INDATA\n/") is None


def test_run_xvmec2000_copies_relative_mgrid_and_cleans_temp(monkeypatch, tmp_path: Path) -> None:
    exec_path = tmp_path / "xvmec2000"
    exec_path.write_text("#!/bin/sh\n")
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n  MGRID_FILE = 'mgrid_case.nc'\n  NITER = 99\n/\n")
    (tmp_path / "mgrid_case.nc").write_text("synthetic mgrid")

    seen: dict[str, object] = {}

    def fake_run(cmd, *, cwd, capture_output, text, timeout, check):
        seen["cmd"] = list(cmd)
        seen["cwd"] = Path(cwd)
        seen["capture_output"] = capture_output
        seen["text"] = text
        seen["timeout"] = timeout
        seen["check"] = check
        assert (Path(cwd) / "input.case").exists()
        assert (Path(cwd) / "mgrid_case.nc").read_text() == "synthetic mgrid"
        assert "NITER = 3" in (Path(cwd) / "input.case").read_text()
        (Path(cwd) / "threed1.case").write_text(
            "  NS =  5 NO. FOURIER MODES =  3 FTOLV =  1.0E-08 NITER =  3\n"
            " ITER FSQR FSQZ FSQL fsqr1 fsqz1 fsql1 DELT0R\n"
            "  1 1.0E-1 2.0E-1 3.0E-1 4.0E-1 5.0E-1 6.0E-1 7.0E-1\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("vmec_jax.vmec2000_exec.subprocess.run", fake_run)

    result = run_xvmec2000(
        input_path,
        exec_path=exec_path,
        workdir=None,
        timeout_s=12.5,
        indata_updates={"niter": "3"},
        keep_workdir=False,
    )

    assert result.returncode == 0
    assert result.stdout == "ok"
    assert len(result.stages) == 1
    assert seen["cmd"] == [str(exec_path), "input.case"]
    assert seen["timeout"] == pytest.approx(12.5)
    assert not Path(result.workdir).exists()
