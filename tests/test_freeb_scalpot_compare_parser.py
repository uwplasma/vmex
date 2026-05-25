from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from tools.diagnostics.vmec2000_exec_freeb_scalpot_compare import (
    _gc_metric_block,
    _missing_vmec_dump_report,
    _missing_required_vmec_dumps,
    main,
    _parse_bextern_dump,
    _parse_fouri_dump,
    _parse_freeb_coupling_dump,
    _parse_gc_dump,
    _parse_scalpot_dump,
    _vmec_run_failure_report,
)


def test_parse_fouri_dump_reads_gsource_source_and_bvecns(tmp_path: Path) -> None:
    p = tmp_path / "fouri_iter53.dat"
    p.write_text(
        "\n".join(
            [
                "# fouri dump",
                "iter2=53",
                "mnpd=3",
                "mnpd2=3",
                "nuv3=4",
                "ndim=1",
                "[gsource]",
                "1  1.0D+00",
                "2  2.0D+00",
                "3  3.0D+00",
                "4  4.0D+00",
                "[source_sym]",
                "1  1.5D+00",
                "2  2.5D+00",
                "3  3.5D+00",
                "4  4.5D+00",
                "[bvecNS]",
                "1  1.0D-01  2.0D-01",
                "2  3.0D-01  4.0D-01",
                "3  5.0D-01  6.0D-01",
            ]
        ),
        encoding="utf-8",
    )

    got = _parse_fouri_dump(p)
    np.testing.assert_allclose(got["gsource"], np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(got["source_sym"], np.array([1.5, 2.5, 3.5, 4.5]))
    np.testing.assert_allclose(got["bvecns_sin"], np.array([0.1, 0.3, 0.5]))
    np.testing.assert_allclose(got["bvecns_cos"], np.array([0.2, 0.4, 0.6]))


def test_parse_scalpot_dump_reads_cached_source_sections(tmp_path: Path) -> None:
    p = tmp_path / "scalpot_iter100_ivacskip2.dat"
    p.write_text(
        "\n".join(
            [
                "# scalpot dump",
                "iter2=100",
                "ivacskip=2",
                "mnpd2=4",
                "mnpd=2",
                "nuv=6",
                "nuv3=4",
                "source_cache_iter=98",
                "[bvec]",
                "1 1.0D+00",
                "2 2.0D+00",
                "3 3.0D+00",
                "4 4.0D+00",
                "[bvecsav]",
                "1 1.0D-01",
                "2 2.0D-01",
                "3 3.0D-01",
                "4 4.0D-01",
                "[source_sym_cached]",
                "1 1.1D+00",
                "2 2.2D+00",
                "3 3.3D+00",
                "4 4.4D+00",
                "[gsource_cached]",
                "1 0.1D+00",
                "2 0.2D+00",
                "3 0.3D+00",
                "4 0.4D+00",
                "5 0.5D+00",
                "6 0.6D+00",
                "[bvecNS_cached]",
                "1 1.0D-02 2.0D-02",
                "2 3.0D-02 4.0D-02",
            ]
        ),
        encoding="utf-8",
    )

    got = _parse_scalpot_dump(p)
    assert got["nuv"] == 6
    assert got["nuv3"] == 4
    assert got["source_cache_iter"] == 98
    np.testing.assert_allclose(got["source_sym_cached"], np.array([1.1, 2.2, 3.3, 4.4]))
    np.testing.assert_allclose(got["gsource_cached"], np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    np.testing.assert_allclose(got["bvecns_cached_sin"], np.array([1.0e-2, 3.0e-2]))
    np.testing.assert_allclose(got["bvecns_cached_cos"], np.array([2.0e-2, 4.0e-2]))


def test_parse_bextern_dump_reads_axis_sections(tmp_path: Path) -> None:
    p = tmp_path / "bextern_iter98.dat"
    p.write_text(
        "\n".join(
            [
                "# bextern dump",
                "iter2=98",
                "nuv3=3",
                "[brad_axis]",
                "1  1.0D-03",
                "2  2.0D-03",
                "3  3.0D-03",
                "[brad_coil]",
                "1  9.0D-03",
                "2  8.0D-03",
                "3  7.0D-03",
                "[bphi_axis]",
                "1 -1.0D-04",
                "2 -2.0D-04",
                "3 -3.0D-04",
                "[bphi_coil]",
                "1 -9.0D-04",
                "2 -8.0D-04",
                "3 -7.0D-04",
                "[bz_axis]",
                "1  4.0D-05",
                "2  5.0D-05",
                "3  6.0D-05",
                "[bz_coil]",
                "1  4.0D-04",
                "2  5.0D-04",
                "3  6.0D-04",
            ]
        ),
        encoding="utf-8",
    )

    got = _parse_bextern_dump(p)
    np.testing.assert_allclose(got["brad_coil"], np.array([9.0e-3, 8.0e-3, 7.0e-3]))
    np.testing.assert_allclose(got["bphi_coil"], np.array([-9.0e-4, -8.0e-4, -7.0e-4]))
    np.testing.assert_allclose(got["bz_coil"], np.array([4.0e-4, 5.0e-4, 6.0e-4]))
    np.testing.assert_allclose(got["brad_axis"], np.array([1.0e-3, 2.0e-3, 3.0e-3]))
    np.testing.assert_allclose(got["bphi_axis"], np.array([-1.0e-4, -2.0e-4, -3.0e-4]))
    np.testing.assert_allclose(got["bz_axis"], np.array([4.0e-5, 5.0e-5, 6.0e-5]))


def test_parse_gc_dump_reads_vmec_layout(tmp_path: Path) -> None:
    p = tmp_path / "gc_raw_ns3_iter10.dat"
    p.write_text(
        "\n".join(
            [
                "# gc dump",
                "ns=3",
                "mpol1=2",
                "ntor=1",
                "ntmax=4",
                "columns: js m n t gcr gcz gcl",
                "1 0 0 1 1.0D+00 2.0D+00 3.0D+00",
                "2 1 1 3 -4.0D+00 5.0D+00 -6.0D+00",
                "3 2 0 4 7.5D-01 -8.5D-01 9.5D-01",
            ]
        ),
        encoding="utf-8",
    )

    got = _parse_gc_dump(p)
    assert got["ns"] == 3
    assert got["mpol1"] == 2
    assert got["ntor"] == 1
    assert got["ntmax"] == 4
    assert got["gcr"].shape == (3, 2, 3, 4)
    np.testing.assert_allclose(got["gcr"][0, 0, 0, 0], 1.0)
    np.testing.assert_allclose(got["gcz"][0, 0, 0, 0], 2.0)
    np.testing.assert_allclose(got["gcl"][0, 0, 0, 0], 3.0)
    np.testing.assert_allclose(got["gcr"][1, 1, 1, 2], -4.0)
    np.testing.assert_allclose(got["gcz"][1, 1, 1, 2], 5.0)
    np.testing.assert_allclose(got["gcl"][1, 1, 1, 2], -6.0)
    np.testing.assert_allclose(got["gcr"][2, 0, 2, 3], 7.5e-1)
    np.testing.assert_allclose(got["gcz"][2, 0, 2, 3], -8.5e-1)
    np.testing.assert_allclose(got["gcl"][2, 0, 2, 3], 9.5e-1)


def test_parse_freeb_coupling_dump_reads_dbsq_and_edge_channels(tmp_path: Path) -> None:
    p = tmp_path / "freeb_coupling_iter8.dat"
    p.write_text(
        "\n".join(
            [
                "# free-boundary coupling dump",
                "iter2=8",
                "presf_ns=1.5D-03",
                "cols: i pgcon rbsq dbsq bsqvac p1e p1o pzu0 pru0",
                "1  1.0D+00  2.0D+00  3.0D+00  4.0D+00  5.0D+00  6.0D+00  7.0D+00  8.0D+00",
                "2  1.5D+00  2.5D+00  3.5D+00  4.5D+00  5.5D+00  6.5D+00  7.5D+00  8.5D+00",
            ]
        ),
        encoding="utf-8",
    )

    got = _parse_freeb_coupling_dump(p)

    assert got["iter2"] == 8
    np.testing.assert_allclose(got["presf_ns"], 1.5e-3)
    np.testing.assert_allclose(got["pgcon"], np.array([1.0, 1.5]))
    np.testing.assert_allclose(got["rbsq"], np.array([2.0, 2.5]))
    np.testing.assert_allclose(got["dbsq"], np.array([3.0, 3.5]))
    np.testing.assert_allclose(got["bsqvac"], np.array([4.0, 4.5]))


def test_gc_metric_block_transposes_jax_gc_axes() -> None:
    vm = np.zeros((2, 2, 3, 4))
    vm[1, 1, 2, 3] = 7.0
    jax = np.transpose(vm, (0, 2, 1, 3))
    got = _gc_metric_block(vm, jax)
    np.testing.assert_allclose(got["rel_raw"], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(got["max_abs"], 0.0, atol=0.0, rtol=0.0)
    assert got["max_loc"] == {"js": 1, "n": 0, "m": 0, "t": 1}


def test_parse_fortran_float_handles_missing_exponent_marker() -> None:
    from tools.diagnostics.vmec2000_exec_freeb_scalpot_compare import _parse_fortran_float

    np.testing.assert_allclose(_parse_fortran_float("1.0D+03"), 1.0e3, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(_parse_fortran_float("1.0564215887228806-316"), 1.0564215887228806e-316, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(_parse_fortran_float("-2.5+02"), -2.5e2, rtol=0.0, atol=0.0)


def test_missing_vmec_dump_report_marks_required_vs_optional(tmp_path: Path) -> None:
    dump_dir = tmp_path / "vmec_dumps"
    dump_dir.mkdir()
    (dump_dir / "bextern_iter80.dat").write_text("# optional dump\n", encoding="utf-8")

    got = _missing_vmec_dump_report(
        vmec_dump_dir=dump_dir,
        iter_target=80,
        vmec_returncodes=[0],
        vmec_exec=tmp_path / "xvmec2000",
        input_path=tmp_path / "input.test",
        workdir=tmp_path,
        missing_required=["scalpot", "vacuum"],
    )

    assert got["status"] == "error"
    assert got["error"]["code"] == "missing_vmec_dumps"
    assert got["error"]["instrumentation_required"] is True
    assert got["error"]["vmec_completed_successfully"] is True
    assert got["error"]["missing_required"] == ["scalpot", "vacuum"]
    assert got["vmec_dump_requirements"]["required"] == ["scalpot", "vacuum"]
    assert "bextern" in got["vmec_dump_requirements"]["optional"]
    assert got["vmec_dump_inventory"]["required"]["scalpot"]["count"] == 0
    assert got["vmec_dump_inventory"]["optional"]["bextern"]["count"] == 1


def test_missing_required_vmec_dumps_requires_scalpot_and_vacuum(tmp_path: Path) -> None:
    dump_dir = tmp_path / "vmec_dumps"
    dump_dir.mkdir()

    assert _missing_required_vmec_dumps(dump_dir, 80) == ["scalpot", "vacuum"]

    (dump_dir / "scalpot_iter80_ivacskip0.dat").write_text("# scalpot\n", encoding="utf-8")
    assert _missing_required_vmec_dumps(dump_dir, 80) == ["vacuum"]

    (dump_dir / "vacuum_iter80_ivacskip0.dat").write_text("# vacuum\n", encoding="utf-8")
    assert _missing_required_vmec_dumps(dump_dir, 80) == []


def test_vmec_run_failure_report_does_not_call_missing_dumps_instrumentation(tmp_path: Path) -> None:
    got = _vmec_run_failure_report(
        vmec_returncodes=[11],
        vmec_exec=tmp_path / "xvmec2000",
        input_path=tmp_path / "input.test",
        workdir=tmp_path,
        vmec_dump_dir=tmp_path / "vmec_dumps",
        iter_target=80,
        missing_required=["vacuum"],
    )

    assert got["status"] == "error"
    assert got["error"]["code"] == "vmec2000_failed"
    assert got["error"]["returncodes"] == [11]
    assert got["error"]["missing_required"] == ["vacuum"]
    assert "instrumentation" in got["error"]["message"]


def test_main_writes_json_for_successful_uninstrumented_vmec(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    input_path = tmp_path / "input.test"
    input_path.write_text("&INDATA\n  MGRID_FILE = 'NONE'\n/\n", encoding="utf-8")
    vmec_exec = tmp_path / "xvmec2000"
    vmec_exec.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    vmec_exec.chmod(0o755)
    workdir = tmp_path / "work"
    json_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vmec2000_exec_freeb_scalpot_compare.py",
            "--input",
            str(input_path),
            "--vmec-exec",
            str(vmec_exec),
            "--iter",
            "80",
            "--max-iter",
            "1",
            "--workdir",
            str(workdir),
            "--json",
            str(json_path),
        ],
    )

    assert main() == 2

    got = json.loads(json_path.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert printed == got
    assert got["error"]["code"] == "missing_vmec_dumps"
    assert got["error"]["missing_required"] == ["scalpot", "vacuum"]
    assert got["vmec_returncodes"] == [0]
    assert not any((workdir / "jax_dumps").iterdir())


def test_main_reports_nonzero_vmec_without_dumps_as_execution_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    input_path = tmp_path / "input.test"
    input_path.write_text("&INDATA\n  MGRID_FILE = 'NONE'\n/\n", encoding="utf-8")
    vmec_exec = tmp_path / "xvmec2000"
    vmec_exec.write_text("#!/bin/sh\nexit 11\n", encoding="utf-8")
    vmec_exec.chmod(0o755)
    workdir = tmp_path / "work"
    json_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vmec2000_exec_freeb_scalpot_compare.py",
            "--input",
            str(input_path),
            "--vmec-exec",
            str(vmec_exec),
            "--iter",
            "80",
            "--max-iter",
            "1",
            "--workdir",
            str(workdir),
            "--json",
            str(json_path),
        ],
    )

    assert main() == 2

    got = json.loads(json_path.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert printed == got
    assert got["error"]["code"] == "vmec2000_failed"
    assert got["error"]["returncodes"] == [11]
    assert got["error"]["missing_required"] == ["scalpot", "vacuum"]
    assert not any((workdir / "jax_dumps").iterdir())
