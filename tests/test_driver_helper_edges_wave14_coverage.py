from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
from vmec_jax.free_boundary import MGridMetadata, PreparedMGrid


class _Input:
    def __init__(self, **values):
        self.values = dict(values)

    def get(self, key, default=None):
        return self.values.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self.values.get(key, default))

    def get_float(self, key, default=0.0):
        return float(self.values.get(key, default))

    def get_int(self, key, default=0):
        return int(self.values.get(key, default))


def test_default_backend_name_falls_back_to_cpu_when_jax_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jax":
            raise RuntimeError("synthetic jax import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert driver._default_backend_name() == "cpu"


def test_dynamic_scan_probe_settings_parses_env_and_clamps_to_available_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "_default_backend_name", lambda: "gpu")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "yes")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "bad-int")

    assert driver._dynamic_scan_probe_settings(5) == (4, True, "gpu")

    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "maybe")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "0")

    assert driver._dynamic_scan_probe_settings(10) == (1, False, "gpu")


def test_list_and_float_normalizers_cover_scalar_iterable_and_invalid_inputs() -> None:
    assert driver._as_float_list(np.asarray(["1.0", "2.5"])) == [1.0, 2.5]
    assert driver._as_float_list(object()) is None
    assert driver._as_list_like((1, 2)) == [1, 2]
    assert driver._as_list_like(np.asarray([3, 4])) == [3, 4]
    assert driver._as_list_like(np.int64(5)) == [np.int64(5)]
    assert driver._as_list_like(v for v in (6, 7)) == [6, 7]


def test_default_use_scan_for_backend_validates_solver_mode_and_gpu_policy() -> None:
    assert driver._default_use_scan_for_backend(_Input(), "cpu", "safe") is False
    assert driver._default_use_scan_for_backend(_Input(), "gpu", "safe") is True
    with pytest.raises(ValueError, match="Unknown solver_mode"):
        driver._default_use_scan_for_backend(_Input(), "cpu", "definitely-not-valid")


def test_resolve_vmec2000_stage_controls_caps_explicit_stage_budgets() -> None:
    niter, ftol, niter_input, ftol_input = driver._resolve_vmec2000_stage_controls(
        nstep=3,
        niter_list=[3, 4, 5],
        ftol_list=[1.0e-3, 1.0e-4, 1.0e-5],
        max_iter=6,
        max_iter_overridden=False,
        multigrid_use_input_niter=True,
        multigrid_user_provided=False,
        accelerated_single_grid_default=False,
        indata=_Input(NITER=99, FTOL=9.0e-9),
    )

    assert niter == [3, 3, 0]
    assert ftol == [1.0e-3, 1.0e-4, 1.0e-5]
    assert niter_input == [3, 4, 5]
    assert ftol_input == [1.0e-3, 1.0e-4, 1.0e-5]


def test_resolve_vmec2000_stage_controls_collapsed_accelerated_single_grid_budget() -> None:
    niter, ftol, niter_input, ftol_input = driver._resolve_vmec2000_stage_controls(
        nstep=1,
        niter_list=[2, 3],
        ftol_list=[1.0e-2, 1.0e-4],
        max_iter=5,
        max_iter_overridden=False,
        multigrid_use_input_niter=True,
        multigrid_user_provided=False,
        accelerated_single_grid_default=True,
        indata=_Input(NITER=99, FTOL=9.0e-9),
    )

    assert niter == [5]
    assert ftol == [1.0e-4]
    assert niter_input is None
    assert ftol_input is None


def test_resolve_vmec2000_stage_controls_distributes_overridden_budget_and_defaults_ftol() -> None:
    niter, ftol, niter_input, ftol_input = driver._resolve_vmec2000_stage_controls(
        nstep=3,
        niter_list=None,
        ftol_list=None,
        max_iter=5,
        max_iter_overridden=True,
        multigrid_use_input_niter=True,
        multigrid_user_provided=False,
        accelerated_single_grid_default=False,
        indata=_Input(NITER=99, FTOL=7.0e-7),
    )

    assert niter == [2, 2, 1]
    assert ftol == [7.0e-7, 7.0e-7, 7.0e-7]
    assert niter_input is None
    assert ftol_input is None


def test_resolve_vmec2000_stage_controls_uses_niter_per_stage_without_input_array() -> None:
    niter, ftol, _, _ = driver._resolve_vmec2000_stage_controls(
        nstep=2,
        niter_list=None,
        ftol_list=None,
        max_iter=9,
        max_iter_overridden=False,
        multigrid_use_input_niter=True,
        multigrid_user_provided=False,
        accelerated_single_grid_default=False,
        indata=_Input(NITER=4, FTOL=8.0e-8),
    )

    assert niter == [4, 4]
    assert ftol == [8.0e-8, 8.0e-8]


def test_resolve_vmec2000_stage_controls_manual_distribution_still_uses_ftol_array() -> None:
    niter, ftol, _, ftol_input = driver._resolve_vmec2000_stage_controls(
        nstep=3,
        niter_list=[10, 20, 30],
        ftol_list=[1.0e-1, 1.0e-2, 1.0e-3],
        max_iter=5,
        max_iter_overridden=False,
        multigrid_use_input_niter=False,
        multigrid_user_provided=True,
        accelerated_single_grid_default=True,
        indata=_Input(NITER=99, FTOL=9.0e-9),
    )

    assert niter == [2, 2, 1]
    assert ftol == [1.0e-1, 1.0e-2, 1.0e-3]
    assert ftol_input == [1.0e-1, 1.0e-2, 1.0e-3]


def test_budget_helpers_cover_empty_stage_and_weight_inputs() -> None:
    assert driver._allocate_integer_budget(total=5, weights=[]) == []
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=7, ns_stages=[]) == 7


def test_sanitize_minimal_resume_state_for_finish_keeps_only_safe_scalars() -> None:
    marker = object()
    assert driver._sanitize_minimal_resume_state_for_finish(marker) is marker

    missing_time = {"inv_tau": [1.0]}
    assert driver._sanitize_minimal_resume_state_for_finish(missing_time) is missing_time

    bad_time = {"time_step": object(), "inv_tau": [1.0]}
    assert driver._sanitize_minimal_resume_state_for_finish(bad_time) is bad_time

    invalid_flip = driver._sanitize_minimal_resume_state_for_finish(
        {"time_step": -0.25, "iter_offset": "3", "vmec2000_cache_valid": 1, "flip_sign": object()}
    )
    assert invalid_flip["time_step"] == pytest.approx(-0.25)
    assert invalid_flip["iter_offset"] == 3
    assert invalid_flip["vmec2000_cache_valid"] is True
    assert invalid_flip["inv_tau"] == [pytest.approx(0.6)] * 10
    assert "flip_sign" not in invalid_flip

    valid = driver._sanitize_minimal_resume_state_for_finish(
        {"time_step": "0.5", "inv_tau": (1.0, 2.0), "iter_offset": 4, "flip_sign": "-1"}
    )
    assert valid == {
        "time_step": 0.5,
        "inv_tau": [1.0, 2.0],
        "iter_offset": 4,
        "vmec2000_cache_valid": False,
        "flip_sign": -1.0,
    }


class _FlakyFloat:
    def __init__(self, value: float):
        self.value = float(value)
        self.calls = 0

    def __float__(self):
        self.calls += 1
        if self.calls == 1:
            raise TypeError("synthetic first conversion failure")
        return self.value


def test_resume_state_sanitizers_cover_conversion_fallbacks_and_missing_time_step() -> None:
    cross = driver._sanitize_resume_state_for_grid_change({"time_step": _FlakyFloat(0.2)}, step_size=0.1)
    same = driver._sanitize_resume_state_for_same_grid({"time_step": _FlakyFloat(0.3)}, step_size=0.1)

    assert cross["time_step"] == pytest.approx(0.2)
    assert cross["inv_tau"] == [pytest.approx(0.75)] * 10
    assert same["time_step"] == pytest.approx(0.3)
    assert same["inv_tau"] == [pytest.approx(0.5)] * 10
    assert driver._sanitize_resume_state_for_grid_change(None, step_size=0.1) is None
    assert driver._sanitize_resume_state_for_same_grid({"inv_tau": [1.0]}, step_size=0.1) is None


def test_result_residual_helpers_cover_bad_payload_fallbacks() -> None:
    assert driver._result_final_residuals(None) is None

    bad_explicit = SimpleNamespace(
        diagnostics={"final_fsqr": object(), "final_fsqz": 2.0, "final_fsql": 3.0},
        fsqr2_history=np.asarray([4.0]),
        fsqz2_history=np.asarray([5.0]),
        fsql2_history=np.asarray([6.0]),
    )
    assert driver._result_final_residuals(bad_explicit) == (4.0, 5.0, 6.0)

    class _BadArray:
        def __array__(self, dtype=None):
            del dtype
            raise TypeError("synthetic array conversion failure")

    bad_diag_histories = SimpleNamespace(
        diagnostics={"fsqr_full": _BadArray(), "fsqz_full": [1.0], "fsql_full": [2.0]},
    )
    assert driver._result_final_residuals(bad_diag_histories) is None
    assert driver._result_final_fsq(SimpleNamespace(diagnostics={"final_fsqr": 1.0, "final_fsqz": 2.0, "final_fsql": 3.0})) == 6.0
    assert driver._result_meets_requested_ftol(None, ftol=1.0) is False
    assert driver._result_meets_requested_ftol(SimpleNamespace(diagnostics={"requested_ftol": 1.0}), ftol=1.0) is False


def test_vmec_history_relerr_reports_scaled_difference_for_matching_shapes() -> None:
    assert driver._vmec_history_relerr(np.asarray([1.0, 1.5]), np.asarray([1.0, 2.0])) == pytest.approx(0.25)


def test_wout_from_fixed_boundary_run_samples_strided_fsqt_and_recomputes_on_bad_fsql(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vmec_jax.wout as wout_module

    captured = {}

    def fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(kind="wout")

    class _BadFinalFsql:
        def __array__(self, dtype=None):
            return np.asarray([[1.0, 2.0]], dtype=dtype)

    result = SimpleNamespace(
        diagnostics={"converged": True},
        fsqr2_history=np.arange(101, dtype=float),
        fsqz2_history=np.arange(101, dtype=float) + 100.0,
        fsql2_history=_BadFinalFsql(),
    )
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=result,
        flux=None,
        profiles={},
        signgs=-1,
    )

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (7.0, 8.0, 9.0))

    out = driver.wout_from_fixed_boundary_run(run, path=tmp_path / "wout_strided.nc")

    assert out.kind == "wout"
    assert captured["fsqr"] == 7.0
    assert captured["fsqz"] == 8.0
    assert captured["fsql"] == 9.0
    assert captured["fsqt"][0] == pytest.approx(102.0)
    assert captured["fsqt"][1] == pytest.approx(106.0)


def test_example_paths_default_root_prefers_reference_then_plain_wout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package_driver = tmp_path / "vmec_jax" / "driver.py"
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    package_driver.parent.mkdir()
    input_path = data_dir / "input.case"
    ref_path = data_dir / "wout_case_reference.nc"
    plain_path = data_dir / "wout_case.nc"
    input_path.write_text("&INDATA\n/\n")
    ref_path.write_text("reference")
    plain_path.write_text("plain")
    monkeypatch.setattr(driver, "__file__", str(package_driver))

    assert driver.example_paths("case") == (input_path, ref_path)

    ref_path.unlink()
    assert driver.example_paths("case") == (input_path, plain_path)


def test_load_example_passes_prepared_mgrid_metadata_without_wout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    input_path = data_dir / "input.freeb"
    input_path.write_text("&INDATA\n/\n")
    cfg = SimpleNamespace(marker="cfg", lfreeb=True)
    indata = object()
    static = object()
    metadata = MGridMetadata(
        path="mgrid.synthetic.nc",
        ir=2,
        jz=3,
        kp=1,
        nfp=1,
        nextcur=1,
        rmin=0.0,
        rmax=1.0,
        zmin=-1.0,
        zmax=1.0,
        mgrid_mode="S",
        coil_groups=("coil",),
        raw_coil_cur=(1.0,),
    )
    prepared = PreparedMGrid(metadata=metadata, extcur=(2.0,))
    captured = {}

    def fake_load_config(path):
        captured["load"] = path
        return cfg, indata

    def fake_prepare_mgrid_for_config(cfg_arg, **kwargs):
        captured["mgrid"] = (cfg_arg, kwargs)
        return prepared

    def fake_build_static(cfg_arg, **kwargs):
        captured["static"] = (cfg_arg, kwargs)
        return static

    monkeypatch.setattr(driver, "load_config", fake_load_config)
    monkeypatch.setattr(driver, "validate_free_boundary_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(driver, "prepare_mgrid_for_config", fake_prepare_mgrid_for_config)
    monkeypatch.setattr(driver, "build_static", fake_build_static)

    loaded = driver.load_example("freeb", root=tmp_path, with_wout=False, grid="grid")

    assert loaded.static is static
    assert loaded.wout is None
    assert loaded.state is None
    assert captured["load"] == str(input_path)
    assert captured["mgrid"] == (cfg, {"load_fields": False, "strict": False})
    assert captured["static"] == (
        cfg,
        {"grid": "grid", "mgrid_metadata": metadata, "free_boundary_extcur": (2.0,)},
    )


def test_save_npz_creates_parent_and_round_trips_arrays(tmp_path: Path) -> None:
    out_path = driver.save_npz(tmp_path / "nested" / "values.npz", values=np.asarray([1.0, 2.0]))

    assert out_path == tmp_path / "nested" / "values.npz"
    with np.load(out_path) as loaded:
        np.testing.assert_allclose(loaded["values"], [1.0, 2.0])
