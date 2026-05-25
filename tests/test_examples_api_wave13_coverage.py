from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import vmec_jax.driver as driver
from vmec_jax.free_boundary import MGridMetadata, PreparedMGrid
from vmec_jax.optimization_workflow import BoundaryModeLimits


ROOT = Path(__file__).resolve().parents[1]
QI_SUPPORT = ROOT / "examples" / "optimization" / "qi_optimization_support.py"


def _load_qi_support_module(name: str = "qi_optimization_support_wave13_test"):
    spec = importlib.util.spec_from_file_location(name, QI_SUPPORT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_paths_prefers_reference_wout_then_default_wout(tmp_path: Path) -> None:
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    input_path = data_dir / "input.demo"
    reference_wout = data_dir / "wout_demo_reference.nc"
    default_wout = data_dir / "wout_demo.nc"
    input_path.write_text("&INDATA\n/\n")
    reference_wout.write_text("reference")
    default_wout.write_text("default")

    found_input, found_wout = driver.example_paths("demo", root=tmp_path)

    assert found_input == input_path
    assert found_wout == reference_wout

    reference_wout.unlink()
    assert driver.example_paths("demo", root=tmp_path) == (input_path, default_wout)


def test_load_example_forwards_prepared_mgrid_metadata_and_extcur(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    input_path = data_dir / "input.freeb"
    input_path.write_text("&INDATA\n/\n")

    cfg = SimpleNamespace(ns=5, lfreeb=True)
    indata = object()
    static = object()
    metadata = MGridMetadata(
        path="mgrid.nc",
        ir=2,
        jz=3,
        kp=4,
        nfp=1,
        nextcur=2,
        rmin=1.0,
        rmax=2.0,
        zmin=-0.5,
        zmax=0.5,
        mgrid_mode="S",
        coil_groups=("A", "B"),
        raw_coil_cur=(10.0, 20.0),
    )
    prepared = PreparedMGrid(metadata=metadata, extcur=(1.5, -2.5))
    calls = {}

    def fake_load_config(path):
        calls["load_config_path"] = path
        return cfg, indata

    def fake_prepare_mgrid_for_config(cfg_arg, **kwargs):
        calls["mgrid"] = (cfg_arg, kwargs)
        return prepared

    monkeypatch.setattr(driver, "load_config", fake_load_config)
    monkeypatch.setattr(driver, "validate_free_boundary_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(driver, "prepare_mgrid_for_config", fake_prepare_mgrid_for_config)

    def fake_build_static(cfg_arg, **kwargs):
        calls["static"] = (cfg_arg, kwargs)
        return static

    monkeypatch.setattr(driver, "build_static", fake_build_static)
    monkeypatch.setattr(driver, "read_wout", lambda _path: (_ for _ in ()).throw(AssertionError("no wout read")))

    loaded = driver.load_example("freeb", root=tmp_path, with_wout=False, grid="tiny-grid")

    assert loaded.input_path == input_path
    assert loaded.wout_path is None
    assert loaded.cfg is cfg
    assert loaded.indata is indata
    assert loaded.static is static
    assert loaded.wout is None
    assert loaded.state is None
    assert calls["load_config_path"] == str(input_path)
    assert calls["mgrid"] == (cfg, {"load_fields": False, "strict": False})
    assert calls["static"] == (
        cfg,
        {
            "grid": "tiny-grid",
            "mgrid_metadata": metadata,
            "free_boundary_extcur": (1.5, -2.5),
        },
    )


def test_qi_support_stage_modes_for_explicit_limits_and_defaults() -> None:
    mod = _load_qi_support_module("qi_optimization_support_stage_modes_wave13_test")
    mod.configure(
        {
            "MAX_MODE": 4,
            "USE_MODE_CONTINUATION": True,
            "CONTINUATION_NFEV": 2,
            "STAGE_REPEATS": 3,
        }
    )

    limits = mod.stage_modes_for(
        {
            "stage_mode_limits": (
                {"mode": 4, "max_m": 1, "max_n": 4, "label": "nfirst"},
                (4, 4),
            )
        }
    )
    explicit_modes = mod.stage_modes_for({"stage_modes": (1, 3, 4)})
    default_modes = mod.stage_modes_for({})

    assert limits == [
        BoundaryModeLimits(mode=4, max_m=1, max_n=4, label="nfirst"),
        BoundaryModeLimits(mode=4, max_m=4, max_n=4),
    ]
    assert explicit_modes == [1, 3, 4]
    assert default_modes == [4, 4, 4]


def test_qi_support_jsonable_and_partial_history_helpers() -> None:
    mod = _load_qi_support_module("qi_optimization_support_jsonable_wave13_test")
    stage_result = SimpleNamespace(
        final_result={
            "_history_dump": {
                "objective_final": np.float64(1.25),
                "qs_final": np.float64(2.5e-3),
                "aspect_final": np.float64(6.1),
                "iota_final": np.float64(0.44),
                "nfev": np.int64(7),
                "total_wall_time_s": np.float64(12.0),
            }
        }
    )

    history = mod._stage_result_history(stage_result)
    partial = mod._partial_diagnostics_from_history(history, {})
    jsonable = mod._jsonable(
        {
            "path": Path("input.demo"),
            "finite": np.float64(3.0),
            "nonfinite": np.asarray([1.0, np.inf]),
            "partial": partial,
        }
    )

    assert history["objective_final"] == np.float64(1.25)
    assert partial == {
        "objective_final": np.float64(1.25),
        "qs_final": np.float64(2.5e-3),
        "aspect": np.float64(6.1),
        "mean_iota": np.float64(0.44),
        "nfev": np.int64(7),
        "total_wall_time_s": np.float64(12.0),
        "partial": True,
        "diagnostics_pending": True,
    }
    assert jsonable == {
        "path": "input.demo",
        "finite": 3.0,
        "nonfinite": [1.0, None],
        "partial": {
            "objective_final": 1.25,
            "qs_final": 2.5e-3,
            "aspect": 6.1,
            "mean_iota": 0.44,
            "nfev": 7,
            "total_wall_time_s": 12.0,
            "partial": True,
            "diagnostics_pending": True,
        },
    }
