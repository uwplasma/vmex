from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.solvers.fixed_boundary.diagnostics.io as dio
import vmec_jax.solvers.fixed_boundary.profiles as sph


def _kernel_terms(ns: int = 3):
    arr = np.arange(float(ns)).reshape(ns, 1, 1) + 1.0
    names = (
        "pr1_even",
        "pr1_odd",
        "pz1_even",
        "pz1_odd",
        "pru_even",
        "pru_odd",
        "pzu_even",
        "pzu_odd",
        "prv_even",
        "prv_odd",
        "pzv_even",
        "pzv_odd",
    )
    return SimpleNamespace(**{name: arr + idx for idx, name in enumerate(names)})


def test_solve_reexports_extracted_helpers():
    import vmec_jax.solve as solve

    assert solve._maybe_dump_time_control_record is dio._maybe_dump_time_control_record
    assert solve._dump_time_control_trace_record is dio._dump_time_control_trace_record
    assert solve._maybe_dump_jacobian_terms_record is dio._maybe_dump_jacobian_terms_record
    assert solve._format_vmec2000_iter_row is dio._format_vmec2000_iter_row
    assert solve._normalize_resume_state_mode is dio._normalize_resume_state_mode
    assert solve._vmec_force_flux_profiles is sph._vmec_force_flux_profiles
    assert solve._mass_half_mesh_from_indata is sph._mass_half_mesh_from_indata
    assert solve._icurv_full_mesh_from_indata is sph._icurv_full_mesh_from_indata


def test_resume_state_modes_and_payloads(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_RESUME_STATE_MODE", " compact ")
    assert dio._normalize_resume_state_mode(None) == "minimal"
    assert dio._normalize_resume_state_mode("light") == "minimal"
    assert dio._normalize_resume_state_mode("off") == "none"
    assert dio._normalize_resume_state_mode("") == "full"

    base = {"iter": 2}
    heavy = {"cache": np.asarray([1.0])}
    assert dio._pack_resume_state_record(base=base, heavy=heavy, mode="minimal") == {"iter": 2}
    full = dio._pack_resume_state_record(base=base, heavy=heavy, mode="full")
    assert full is not base
    assert full["iter"] == 2
    np.testing.assert_allclose(full["cache"], [1.0])
    assert dio._pack_resume_state_record(base=base, heavy=heavy, mode="none") is None

    with pytest.raises(ValueError, match="resume_state_mode"):
        dio._normalize_resume_state_mode("invalid")


def test_adjoint_trace_mode_aliases_and_materialization():
    assert dio._normalize_adjoint_trace_mode("") == "full"
    assert dio._normalize_adjoint_trace_mode("full") == "full"
    assert dio._normalize_adjoint_trace_mode("dynamic") == "dynamic"
    assert dio._normalize_adjoint_trace_mode("branch") == "branch"
    assert dio._normalize_adjoint_trace_mode("branch_local") == "branch"
    assert dio._normalize_adjoint_trace_mode("lean") == "branch"
    assert dio._normalize_adjoint_trace_mode("compact") == "branch"

    value = np.asarray([1.0, 2.0])
    assert dio._materialize_adjoint_trace_array(value, mode="dynamic") is value
    np.testing.assert_allclose(dio._materialize_adjoint_trace_array(value, mode="branch"), value)

    with pytest.raises(ValueError, match="adjoint_trace_mode"):
        dio._normalize_adjoint_trace_mode("invalid")


def test_print_cadence_and_formatters_preserve_legacy_text():
    assert dio._vmec2000_cadence_selected(iter_idx=1, max_iter=10, nstep_screen=99)
    assert dio._vmec2000_cadence_selected(iter_idx=10, max_iter=10, nstep_screen=99)
    assert dio._vmec2000_cadence_selected(iter_idx=4, max_iter=10, nstep_screen=2)
    assert dio._vmec2000_cadence_selected(iter_idx=4, max_iter=10, nstep_screen=0)
    assert not dio._should_print_vmec2000_row(
        iter_idx=4,
        max_iter=10,
        nstep_screen=2,
        verbose=False,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
    )

    sym = dio._format_vmec2000_iter_row(
        iter_idx=3,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=4.0,
        r00=5.0,
        w_mhd=6.0,
        lasym=False,
    )
    asym = dio._format_vmec2000_iter_row(
        iter_idx=3,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=4.0,
        r00=5.0,
        z00=None,
        w_mhd=6.0,
        lasym=True,
    )
    assert "  1.00E+00" in sym
    assert "NAN" in asym.upper()
    assert dio._format_axis_coeff(1.0e-6) == "1E-06"
    assert dio._format_time_control_log_row(iter_idx=8, fsq=1, fsq0=2, res0=3, res1=4, time_step=0.5).endswith(
        "time_step=5.000000e-01\n"
    )
    assert dio._format_time_control_trace_row(
        stage="restart",
        iter2=2,
        iter1=1,
        fsq=1,
        fsq0=2,
        res0=3,
        res1=4,
        time_step=0.5,
        irst=2,
    ).endswith("   2 restart\n")
    assert dio._format_checkpoint_log_row(iter_idx=8, fsq=1, fsq0=2, res0=3, res1=4).startswith("iter=8")
    assert dio._format_freeb_control_trace_row(
        iter2=2,
        iter1=1,
        ivac=3,
        ivacskip=4,
        nvacskip=5,
        fsq_rz_prev=0.25,
        cached=False,
    ).endswith(" 0\n")
    assert dio._format_evolve_trace_row(
        iter2=2,
        iter1=1,
        ns=3,
        stage="pre",
        fsq1=1,
        fsq_prev=2,
        time_step=0.5,
        dtau=0.25,
        b1=0.75,
        fac=0.8,
        xc_norm=10,
        v_norm=11,
        g_norm=12,
    ).startswith("       2        1        3 pre")


def test_legacy_dump_path_and_iter_filter_truthiness(monkeypatch, tmp_path):
    monkeypatch.delenv("VMEC_JAX_DUMP_DIR", raising=False)
    monkeypatch.setenv("VMEC_JAX_DUMP_TIMECONTROL", "1")
    assert dio._legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_TIMECONTROL", filename="x.log") is None

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_TIMECONTROL", "")
    assert dio._legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_TIMECONTROL", filename="x.log") is None
    monkeypatch.setenv("VMEC_JAX_DUMP_TIMECONTROL", "0")
    assert dio._legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_TIMECONTROL", filename="x.log") is None

    monkeypatch.setenv("VMEC_JAX_DUMP_TIMECONTROL", "false")
    assert dio._legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_TIMECONTROL", filename="x.log") == tmp_path / "x.log"
    monkeypatch.setenv("VMEC_JAX_DUMP_TIMECONTROL", " 0 ")
    assert dio._legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_TIMECONTROL", filename="x.log") == tmp_path / "x.log"

    assert dio._legacy_single_dump_iter_selected(dump_iter="", iter_idx=3)
    assert dio._legacy_single_dump_iter_selected(dump_iter="3", iter_idx=3)
    assert not dio._legacy_single_dump_iter_selected(dump_iter="2", iter_idx=3)
    assert dio._legacy_single_dump_iter_selected(dump_iter="not-an-int", iter_idx=3)


def test_record_writers_append_expected_files(monkeypatch, tmp_path):
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_TIMECONTROL", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_CHECKPOINT", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_FREEB_CONTROL", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_FREEB_AXIS", "1")

    dio._maybe_dump_time_control_record(iter_idx=2, fsq=1, fsq0=2, res0=3, res1=4, time_step=0.5)
    dio._dump_time_control_trace_record(
        stage="restart",
        iter2=2,
        iter1=1,
        fsq=1,
        fsq0=2,
        res0=3,
        res1=4,
        time_step=0.5,
        irst=2,
    )
    dio._maybe_dump_checkpoint_record(iter_idx=2, fsq=1, fsq0=2, res0=3, res1=4)
    dio._dump_freeb_control_trace_record(
        iter2=2,
        iter1=1,
        ivac=3,
        ivacskip=4,
        nvacskip=5,
        fsq_rz_prev=0.25,
        cached=True,
    )
    dio._dump_freeb_axis_trace_record(iter2=2, axis_r=np.array([[1.0, 2.0]]), axis_z=np.array([[3.0, 4.0]]))

    assert "time_step=5.000000e-01" in (tmp_path / "time_control.log").read_text(encoding="utf-8")
    assert (tmp_path / "time_control_trace.log").read_text(encoding="utf-8").endswith("   2 restart\n")
    assert "iter=2 fsq=1.000000e+00" in (tmp_path / "checkpoint.log").read_text(encoding="utf-8")
    assert (tmp_path / "freeb_control_trace.log").read_text(encoding="utf-8").endswith(" 1\n")
    with np.load(tmp_path / "freeb_axis_iter2.npz") as data:
        np.testing.assert_allclose(data["axis_r"], [1.0, 2.0])
        np.testing.assert_allclose(data["axis_z"], [3.0, 4.0])


def test_record_write_failures_are_nonfatal(monkeypatch):
    class BadPath:
        def open(self, *args, **kwargs):
            raise OSError("blocked")

    monkeypatch.setattr(dio, "_legacy_dump_record_path", lambda **kwargs: BadPath())

    dio._maybe_dump_time_control_record(iter_idx=1, fsq=1, fsq0=2, res0=3, res1=4, time_step=0.5)
    dio._dump_time_control_trace_record(stage="s", iter2=1, iter1=1, fsq=1, fsq0=2, res0=3, res1=4, time_step=0.5, irst=1)
    dio._maybe_dump_checkpoint_record(iter_idx=1, fsq=1, fsq0=2, res0=3, res1=4)
    dio._dump_freeb_control_trace_record(iter2=1, iter1=1, ivac=1, ivacskip=0, nvacskip=2, fsq_rz_prev=0.25, cached=True)
    dio._dump_freeb_axis_trace_record(iter2=1, axis_r=np.array([1.0]), axis_z=np.array([2.0]))


def test_evolve_trace_writer_materializes_optional_blocks(monkeypatch, tmp_path):
    import vmec_jax.diagnostics as diagnostics

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_EVOLVE", "1")
    static = SimpleNamespace(cfg=SimpleNamespace(ns=2))

    def fake_internal_mn_from_state(*args, **kwargs):
        return {
            "rcc": np.array([1.0]),
            "rss": np.array([2.0]),
            "zsc": np.array([3.0]),
            "zcs": np.array([4.0]),
            "lsc": np.array([5.0]),
            "lcs": np.array([6.0]),
            "rsc": np.array([7.0]),
            "rcs": np.array([8.0]),
            "zcc": np.array([9.0]),
            "zss": np.array([10.0]),
            "lcc": np.array([11.0]),
            "lss": np.array([12.0]),
        }

    def fake_xc_from_mn_blocks(*, cfg, **kwargs):
        assert cfg is static.cfg
        return np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in kwargs.values() if value is not None])

    monkeypatch.setattr(diagnostics, "vmec_internal_mn_from_state", fake_internal_mn_from_state)
    monkeypatch.setattr(diagnostics, "vmec_xc_from_mn_blocks", fake_xc_from_mn_blocks)

    dio._maybe_dump_evolve_trace_record(
        static=static,
        iter2=4,
        iter1=1,
        stage="post",
        fsq1_val=1.0,
        fsq_prev_val=2.0,
        time_step_val=0.5,
        dtau_val=0.25,
        b1_val=0.75,
        fac_val=0.5,
        state_val=object(),
        vRcc_val=np.array([1.0]),
        vRss_val=np.array([2.0]),
        vZsc_val=np.array([3.0]),
        vZcs_val=np.array([4.0]),
        vLsc_val=np.array([5.0]),
        vLcs_val=np.array([6.0]),
        vRsc_val=np.array([7.0]),
        vRcs_val=np.array([8.0]),
        vZcc_val=np.array([9.0]),
        vZss_val=np.array([10.0]),
        vLcc_val=np.array([11.0]),
        vLss_val=np.array([12.0]),
        frcc_val=np.array([2.0]),
        frss_val=np.array([3.0]),
        fzsc_val=np.array([4.0]),
        fzcs_val=np.array([5.0]),
        flsc_val=np.array([6.0]),
        flcs_val=np.array([7.0]),
        frsc_val=np.array([8.0]),
        frcs_val=np.array([9.0]),
        fzcc_val=np.array([10.0]),
        fzss_val=np.array([11.0]),
        flcc_val=np.array([12.0]),
        flss_val=np.array([13.0]),
    )

    fields = (tmp_path / "evolve_trace.log").read_text(encoding="utf-8").split()
    assert fields[:4] == ["4", "1", "2", "post"]
    assert float(fields[-1]) == pytest.approx(np.linalg.norm(np.arange(2.0, 14.0)))


def test_jacobian_terms_dump_preserves_filter_and_false_token_quirks(monkeypatch, tmp_path):
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_JACOBIAN_TERMS", "false")
    dio._maybe_dump_jacobian_terms_record(k=_kernel_terms(), s=np.array([0.0, 0.5, 1.0]), iter_idx=3)
    assert not (tmp_path / "jacobian_terms_iter3.dat").exists()

    monkeypatch.setenv("VMEC_JAX_DUMP_JACOBIAN_TERMS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "4")
    dio._maybe_dump_jacobian_terms_record(k=_kernel_terms(), s=np.array([0.0, 0.5, 1.0]), iter_idx=3)
    assert not (tmp_path / "jacobian_terms_iter3.dat").exists()

    monkeypatch.setenv("VMEC_JAX_DUMP_JACOBIAN_TERMS", "FALSE")
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "not-an-int")
    dio._maybe_dump_jacobian_terms_record(k=_kernel_terms(), s=np.array([0.0, 0.5, 1.0]), iter_idx=3)

    text = (tmp_path / "jacobian_terms_iter3.dat").read_text(encoding="utf-8")
    assert "ru12 pzs pzu12 prs pr12 ptau" in text
    assert len([line for line in text.splitlines() if line.startswith("     ")]) == 2


def test_scalar_and_adjoint_trace_materialization():
    assert dio._finite_float_or_zero(np.array(3.25)) == 3.25
    assert dio._finite_float_or_zero(np.nan) == 0.0
    assert dio._finite_float_or_zero(np.inf) == 0.0
    assert dio._normalize_adjoint_trace_mode(" dynamic ") == "dynamic"

    value = object()
    assert dio._materialize_adjoint_trace_array(value, mode="dynamic") is value
    np.testing.assert_allclose(dio._materialize_adjoint_trace_array([1.0, 2.0], mode="FULL"), [1.0, 2.0])

    with pytest.raises(ValueError, match="adjoint_trace_mode"):
        dio._normalize_adjoint_trace_mode("summary")
